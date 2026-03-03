use crate::{Spi, SpiMode};
use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::thread;
use std::time::Duration;

// --- GPIO chardev v2 (output-only) ----------------------------------------

const GPIO_V2_LINE_FLAG_OUTPUT: u64 = 1 << 1;

const fn ioc(dir: u8, ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ((dir as libc::c_ulong) << 30)
        | ((ty as libc::c_ulong) << 8)
        | (nr as libc::c_ulong)
        | ((size as libc::c_ulong) << 16)
}

const fn iowr(ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ioc(3, ty, nr, size)
}

#[repr(C)]
struct GpioV2LineAttribute {
    id: u32,
    _pad: u32,
    data: u64, // union: flags / values / debounce_period_us
}

#[repr(C)]
struct GpioV2LineConfigAttr {
    attr: GpioV2LineAttribute,
    mask: u64,
}

#[repr(C)]
struct GpioV2LineConfig {
    flags: u64,
    num_attrs: u32,
    _pad: [u32; 5],
    attrs: [GpioV2LineConfigAttr; 10],
}

#[repr(C)]
struct GpioV2LineRequest {
    offsets: [u32; 64],
    consumer: [u8; 32],
    config: GpioV2LineConfig,
    num_lines: u32,
    event_buffer_size: u32,
    _pad: [u32; 5],
    fd: i32,
}

#[repr(C)]
struct GpioV2LineValues {
    bits: u64,
    mask: u64,
}

const GPIO_V2_GET_LINE: libc::c_ulong =
    iowr(0xB4, 0x07, size_of::<GpioV2LineRequest>());
const GPIO_V2_LINE_SET_VALUES: libc::c_ulong =
    iowr(0xB4, 0x0D, size_of::<GpioV2LineValues>());

/// A single GPIO output line via the Linux chardev v2 interface.
pub struct GpioOutput {
    fd: OwnedFd,
}

impl GpioOutput {
    /// Request line `line` as output on `/dev/gpiochipN`.
    pub fn new(chip: u32, line: u32) -> io::Result<Self> {
        let path = format!("/dev/gpiochip{chip}");
        let chip_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;

        let mut req: GpioV2LineRequest = unsafe { std::mem::zeroed() };
        req.offsets[0] = line;
        req.num_lines = 1;
        req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        req.consumer[..5].copy_from_slice(b"actor");

        if unsafe { libc::ioctl(chip_file.as_raw_fd(), GPIO_V2_GET_LINE, &mut req) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            fd: unsafe { OwnedFd::from_raw_fd(req.fd) },
        })
    }

    /// Drive the line high (`true`) or low (`false`).
    pub fn set(&self, high: bool) -> io::Result<()> {
        let vals = GpioV2LineValues {
            bits: u64::from(high),
            mask: 1,
        };
        if unsafe { libc::ioctl(self.fd.as_raw_fd(), GPIO_V2_LINE_SET_VALUES, &vals) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

// --- ST7789 commands -------------------------------------------------------

const SWRESET: u8 = 0x01;
const SLPOUT: u8 = 0x11;
const INVOFF: u8 = 0x20;
const INVON: u8 = 0x21;
const DISPON: u8 = 0x29;
const CASET: u8 = 0x2A;
const RASET: u8 = 0x2B;
const RAMWR: u8 = 0x2C;
const MADCTL: u8 = 0x36;
const COLMOD: u8 = 0x3A;
const FRMCTR2: u8 = 0xB2;
const GCTRL: u8 = 0xB7;
const VCOMS: u8 = 0xBB;
const LCMCTRL: u8 = 0xC0;
const VDVVRHEN: u8 = 0xC2;
const VRHS: u8 = 0xC3;
const VDVS: u8 = 0xC4;
const FRCTRL2: u8 = 0xC6;
const GMCTRP1: u8 = 0xE0;
const GMCTRN1: u8 = 0xE1;
const PWCTRL1: u8 = 0xD0;

// --- Public types ----------------------------------------------------------

/// Display rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    Deg0,
    Deg90,
    Deg180,
    Deg270,
}

impl Rotation {
    fn madctl(self) -> u8 {
        match self {
            Self::Deg0 => 0x00,
            Self::Deg90 => 0x60,
            Self::Deg180 => 0xC0,
            Self::Deg270 => 0xA0,
        }
    }

    fn swaps(self) -> bool {
        matches!(self, Self::Deg90 | Self::Deg270)
    }
}

/// Configuration for [`St7789`].
pub struct St7789Config {
    pub width: u16,
    pub height: u16,
    pub rotation: Rotation,
    pub invert: bool,
    pub spi_speed_hz: u32,
    pub offset_left: u16,
    pub offset_top: u16,
}

impl Default for St7789Config {
    fn default() -> Self {
        Self {
            width: 240,
            height: 240,
            rotation: Rotation::Deg0,
            invert: true,
            spi_speed_hz: 4_000_000,
            offset_left: 0,
            offset_top: 0,
        }
    }
}

/// ST7789 TFT LCD driver over SPI with GPIO control lines.
pub struct St7789 {
    spi: Spi,
    dc: GpioOutput,
    rst: Option<GpioOutput>,
    bl: Option<GpioOutput>,
    raw_width: u16,
    raw_height: u16,
    rotation: Rotation,
    invert: bool,
    offset_left: u16,
    offset_top: u16,
}

impl St7789 {
    /// Initialise the display.  Configures SPI, runs the hardware-reset
    /// sequence (when `rst` is provided), and sends the full init commands.
    pub fn new(
        spi: Spi,
        dc: GpioOutput,
        rst: Option<GpioOutput>,
        bl: Option<GpioOutput>,
        config: St7789Config,
    ) -> io::Result<Self> {
        spi.set_mode(SpiMode::Mode0)?;
        spi.set_lsb_first(false)?;
        spi.set_max_speed_hz(config.spi_speed_hz)?;

        let disp = Self {
            spi,
            dc,
            rst,
            bl,
            raw_width: config.width,
            raw_height: config.height,
            rotation: config.rotation,
            invert: config.invert,
            offset_left: config.offset_left,
            offset_top: config.offset_top,
        };

        if let Some(ref bl) = disp.bl {
            bl.set(false)?;
            thread::sleep(Duration::from_millis(100));
            bl.set(true)?;
        }

        disp.reset()?;
        disp.init_seq()?;
        Ok(disp)
    }

    // -- low-level ----------------------------------------------------------

    /// Send a single command byte (DC low).
    pub fn command(&self, cmd: u8) -> io::Result<()> {
        self.dc.set(false)?;
        self.spi.write(&[cmd])
    }

    /// Send data bytes (DC high), chunked to 4 KiB per SPI transfer.
    pub fn data(&self, buf: &[u8]) -> io::Result<()> {
        self.dc.set(true)?;
        for chunk in buf.chunks(4096) {
            self.spi.write(chunk)?;
        }
        Ok(())
    }

    // -- public API ---------------------------------------------------------

    /// Hardware-reset the display (no-op when no reset pin was provided).
    pub fn reset(&self) -> io::Result<()> {
        if let Some(ref rst) = self.rst {
            rst.set(true)?;
            thread::sleep(Duration::from_millis(500));
            rst.set(false)?;
            thread::sleep(Duration::from_millis(500));
            rst.set(true)?;
            thread::sleep(Duration::from_millis(500));
        }
        Ok(())
    }

    /// Turn the backlight on or off (no-op when no backlight pin was provided).
    pub fn set_backlight(&self, on: bool) -> io::Result<()> {
        if let Some(ref bl) = self.bl {
            bl.set(on)?;
        }
        Ok(())
    }

    /// Effective display width after rotation.
    pub fn width(&self) -> u16 {
        if self.rotation.swaps() { self.raw_height } else { self.raw_width }
    }

    /// Effective display height after rotation.
    pub fn height(&self) -> u16 {
        if self.rotation.swaps() { self.raw_width } else { self.raw_height }
    }

    /// Set the pixel-address window for subsequent RAM writes.
    pub fn set_window(&self, x0: u16, y0: u16, x1: u16, y1: u16) -> io::Result<()> {
        let x0 = x0 + self.offset_left;
        let x1 = x1 + self.offset_left;
        let y0 = y0 + self.offset_top;
        let y1 = y1 + self.offset_top;

        self.command(CASET)?;
        self.data(&[(x0 >> 8) as u8, x0 as u8, (x1 >> 8) as u8, x1 as u8])?;
        self.command(RASET)?;
        self.data(&[(y0 >> 8) as u8, y0 as u8, (y1 >> 8) as u8, y1 as u8])?;
        self.command(RAMWR)
    }

    /// Write raw RGB565 big-endian pixel data covering the full display.
    ///
    /// `data` must be exactly `width() * height() * 2` bytes.
    pub fn display(&self, data: &[u8]) -> io::Result<()> {
        self.set_window(0, 0, self.width() - 1, self.height() - 1)?;
        self.data(data)
    }

    /// Convert an RGB888 buffer to RGB565 and write it to the full display.
    ///
    /// `rgb` must be exactly `width() * height() * 3` bytes (R, G, B per
    /// pixel, row-major).
    pub fn display_rgb888(&self, rgb: &[u8]) -> io::Result<()> {
        self.set_window(0, 0, self.width() - 1, self.height() - 1)?;
        self.dc.set(true)?;
        let mut buf = [0u8; 4096];
        let mut pos = 0;
        for px in rgb.chunks_exact(3) {
            let [hi, lo] = rgb565(px[0], px[1], px[2]);
            buf[pos] = hi;
            buf[pos + 1] = lo;
            pos += 2;
            if pos + 1 >= buf.len() {
                self.spi.write(&buf[..pos])?;
                pos = 0;
            }
        }
        if pos > 0 {
            self.spi.write(&buf[..pos])?;
        }
        Ok(())
    }

    // -- init sequence (matches pimoroni/st7789-python) ---------------------

    fn init_seq(&self) -> io::Result<()> {
        self.command(SWRESET)?;
        thread::sleep(Duration::from_millis(150));

        self.command(MADCTL)?;
        self.data(&[self.rotation.madctl()])?;

        self.command(FRMCTR2)?;
        self.data(&[0x0C, 0x0C, 0x00, 0x33, 0x33])?;

        self.command(COLMOD)?;
        self.data(&[0x05])?; // 16-bit RGB565

        self.command(GCTRL)?;
        self.data(&[0x14])?;

        self.command(VCOMS)?;
        self.data(&[0x37])?;

        self.command(LCMCTRL)?;
        self.data(&[0x2C])?;

        self.command(VDVVRHEN)?;
        self.data(&[0x01])?;

        self.command(VRHS)?;
        self.data(&[0x12])?;

        self.command(VDVS)?;
        self.data(&[0x20])?;

        self.command(PWCTRL1)?;
        self.data(&[0xA4, 0xA1])?;

        self.command(FRCTRL2)?;
        self.data(&[0x0F])?;

        self.command(GMCTRP1)?;
        self.data(&[
            0xD0, 0x04, 0x0D, 0x11, 0x13, 0x2B, 0x3F, 0x54,
            0x4C, 0x18, 0x0D, 0x0B, 0x1F, 0x23,
        ])?;

        self.command(GMCTRN1)?;
        self.data(&[
            0xD0, 0x04, 0x0C, 0x11, 0x13, 0x2C, 0x3F, 0x44,
            0x51, 0x2F, 0x1F, 0x1F, 0x20, 0x23,
        ])?;

        self.command(if self.invert { INVON } else { INVOFF })?;

        self.command(SLPOUT)?;
        self.command(DISPON)?;
        thread::sleep(Duration::from_millis(100));

        Ok(())
    }
}

/// Convert an (R, G, B) triplet to big-endian RGB565.
pub fn rgb565(r: u8, g: u8, b: u8) -> [u8; 2] {
    let v = ((r as u16 & 0xF8) << 8) | ((g as u16 & 0xFC) << 3) | (b as u16 >> 3);
    v.to_be_bytes()
}
