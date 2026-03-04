use std::{
    fs::{File, OpenOptions},
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    thread,
    time::Duration,
};

const SPI_IOC_MAGIC: u8 = b'k';

const fn ior(ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ioc(2, ty, nr, size)
}

const fn iow(ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ioc(1, ty, nr, size)
}

const SPI_IOC_RD_MODE: libc::c_ulong = ior(SPI_IOC_MAGIC, 1, 1);
const SPI_IOC_WR_MODE: libc::c_ulong = iow(SPI_IOC_MAGIC, 1, 1);
const SPI_IOC_RD_LSB_FIRST: libc::c_ulong = ior(SPI_IOC_MAGIC, 2, 1);
const SPI_IOC_WR_LSB_FIRST: libc::c_ulong = iow(SPI_IOC_MAGIC, 2, 1);
const SPI_IOC_RD_BITS_PER_WORD: libc::c_ulong = ior(SPI_IOC_MAGIC, 3, 1);
const SPI_IOC_WR_BITS_PER_WORD: libc::c_ulong = iow(SPI_IOC_MAGIC, 3, 1);
const SPI_IOC_RD_MAX_SPEED_HZ: libc::c_ulong = ior(SPI_IOC_MAGIC, 4, 4);
const SPI_IOC_WR_MAX_SPEED_HZ: libc::c_ulong = iow(SPI_IOC_MAGIC, 4, 4);

fn spi_ioc_message(n: usize) -> libc::c_ulong {
    iow(SPI_IOC_MAGIC, 0, n * size_of::<SpiIocTransfer>())
}

#[repr(C)]
struct SpiIocTransfer {
    tx_buf: u64,
    rx_buf: u64,
    len: u32,
    speed_hz: u32,
    delay_usecs: u16,
    bits_per_word: u8,
    cs_change: u8,
    tx_nbits: u8,
    rx_nbits: u8,
    word_delay_usecs: u8,
    _pad: u8,
}

impl SpiIocTransfer {
    fn zero(len: u32) -> Self {
        Self {
            tx_buf: 0,
            rx_buf: 0,
            len,
            speed_hz: 0,
            delay_usecs: 0,
            bits_per_word: 0,
            cs_change: 0,
            tx_nbits: 0,
            rx_nbits: 0,
            word_delay_usecs: 0,
            _pad: 0,
        }
    }
}

/// SPI clock mode (CPOL, CPHA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpiMode {
    Mode0 = 0,
    Mode1 = 0x01,
    Mode2 = 0x02,
    Mode3 = 0x03,
}

impl TryFrom<u8> for SpiMode {
    type Error = io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x03 {
            0 => Ok(Self::Mode0),
            1 => Ok(Self::Mode1),
            2 => Ok(Self::Mode2),
            3 => Ok(Self::Mode3),
            _ => unreachable!(),
        }
    }
}

/// Linux SPI device handle wrapping `/dev/spidevB.D`.
pub struct Spi {
    file: File,
}

impl Spi {
    /// Open an SPI device by bus and chip-select number.
    pub fn open(bus: u8, device: u8) -> io::Result<Self> {
        let path = format!("/dev/spidev{bus}.{device}");
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        Ok(Self { file })
    }

    fn fd(&self) -> libc::c_int {
        self.file.as_raw_fd()
    }

    fn ioctl_rd_u8(&self, request: libc::c_ulong) -> io::Result<u8> {
        let mut val: u8 = 0;
        if unsafe { libc::ioctl(self.fd(), request, &mut val) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(val)
    }

    fn ioctl_wr_u8(&self, request: libc::c_ulong, val: u8) -> io::Result<()> {
        if unsafe { libc::ioctl(self.fd(), request, &val) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn ioctl_rd_u32(&self, request: libc::c_ulong) -> io::Result<u32> {
        let mut val: u32 = 0;
        if unsafe { libc::ioctl(self.fd(), request, &mut val) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(val)
    }

    fn ioctl_wr_u32(&self, request: libc::c_ulong, val: u32) -> io::Result<()> {
        if unsafe { libc::ioctl(self.fd(), request, &val) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn mode(&self) -> io::Result<SpiMode> {
        self.ioctl_rd_u8(SPI_IOC_RD_MODE)?.try_into()
    }

    pub fn set_mode(&self, mode: SpiMode) -> io::Result<()> {
        self.ioctl_wr_u8(SPI_IOC_WR_MODE, mode as u8)
    }

    pub fn bits_per_word(&self) -> io::Result<u8> {
        self.ioctl_rd_u8(SPI_IOC_RD_BITS_PER_WORD)
    }

    pub fn set_bits_per_word(&self, bits: u8) -> io::Result<()> {
        self.ioctl_wr_u8(SPI_IOC_WR_BITS_PER_WORD, bits)
    }

    pub fn max_speed_hz(&self) -> io::Result<u32> {
        self.ioctl_rd_u32(SPI_IOC_RD_MAX_SPEED_HZ)
    }

    pub fn set_max_speed_hz(&self, speed: u32) -> io::Result<()> {
        self.ioctl_wr_u32(SPI_IOC_WR_MAX_SPEED_HZ, speed)
    }

    pub fn lsb_first(&self) -> io::Result<bool> {
        Ok(self.ioctl_rd_u8(SPI_IOC_RD_LSB_FIRST)? != 0)
    }

    pub fn set_lsb_first(&self, lsb: bool) -> io::Result<()> {
        self.ioctl_wr_u8(SPI_IOC_WR_LSB_FIRST, lsb as u8)
    }

    /// Full-duplex transfer: sends `tx` while simultaneously receiving into `rx`.
    /// Both slices must have the same length.
    pub fn transfer(&self, tx: &[u8], rx: &mut [u8]) -> io::Result<()> {
        assert_eq!(tx.len(), rx.len(), "tx and rx must be the same length");
        let mut xfer = SpiIocTransfer::zero(tx.len() as u32);
        xfer.tx_buf = tx.as_ptr() as u64;
        xfer.rx_buf = rx.as_mut_ptr() as u64;
        if unsafe { libc::ioctl(self.fd(), spi_ioc_message(1), &xfer) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Write bytes to the SPI device (TX only, no read).
    pub fn write(&self, data: &[u8]) -> io::Result<()> {
        let mut xfer = SpiIocTransfer::zero(data.len() as u32);
        xfer.tx_buf = data.as_ptr() as u64;
        if unsafe { libc::ioctl(self.fd(), spi_ioc_message(1), &xfer) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Read bytes from the SPI device (clocks out zeros while reading).
    pub fn read(&self, buf: &mut [u8]) -> io::Result<()> {
        let mut xfer = SpiIocTransfer::zero(buf.len() as u32);
        xfer.rx_buf = buf.as_mut_ptr() as u64;
        if unsafe { libc::ioctl(self.fd(), spi_ioc_message(1), &xfer) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Perform multiple transfers in a single SPI transaction (CS held low
    /// for the entire sequence). Per-transfer speed and bits-per-word
    /// overrides are supported; set to 0 for the device default.
    pub fn transfer_many(&self, transfers: &mut [SpiTransfer<'_>]) -> io::Result<()> {
        let ioc: Vec<SpiIocTransfer> = transfers
            .iter_mut()
            .map(|t| {
                let tx_ptr = t.tx.map_or(std::ptr::null(), |b| b.as_ptr());
                let rx_ptr = t.rx.as_deref_mut().map_or(std::ptr::null_mut(), |b| b.as_mut_ptr());
                let len = t.tx.map_or(0, |b| b.len()).max(t.rx.as_deref().map_or(0, |b| b.len()));
                let mut x = SpiIocTransfer::zero(len as u32);
                x.tx_buf = tx_ptr as u64;
                x.rx_buf = rx_ptr as u64;
                x.speed_hz = t.speed_hz;
                x.delay_usecs = t.delay_usecs;
                x.bits_per_word = t.bits_per_word;
                x
            })
            .collect();
        if unsafe { libc::ioctl(self.fd(), spi_ioc_message(ioc.len()), ioc.as_ptr()) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// A single segment in a multi-transfer SPI transaction.
pub struct SpiTransfer<'a> {
    pub tx: Option<&'a [u8]>,
    pub rx: Option<&'a mut [u8]>,
    /// Per-transfer speed override (0 = device default).
    pub speed_hz: u32,
    /// Delay after this segment in microseconds.
    pub delay_usecs: u16,
    /// Per-transfer bits-per-word override (0 = device default).
    pub bits_per_word: u8,
}

const GPIO_V2_LINE_FLAG_OUTPUT: u64 = 1 << 3;

const fn ioc(dir: u8, ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ((dir as libc::c_ulong) << 30) | ((ty as libc::c_ulong) << 8) | (nr as libc::c_ulong) | ((size as libc::c_ulong) << 16)
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

const GPIO_V2_GET_LINE: libc::c_ulong = iowr(0xB4, 0x07, size_of::<GpioV2LineRequest>());
const GPIO_V2_LINE_SET_VALUES: libc::c_ulong = iowr(0xB4, 0x0F, size_of::<GpioV2LineValues>());

/// A single GPIO output line via the Linux chardev v2 interface.
pub struct GpioOutput {
    fd: OwnedFd,
}

impl GpioOutput {
    /// Request line `line` as output on `/dev/gpiochipN`.
    pub fn new(chip: u32, line: u32) -> io::Result<Self> {
        let path = format!("/dev/gpiochip{chip}");
        let chip_file = OpenOptions::new().read(true).write(true).open(&path)?;

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
        if self.rotation.swaps() {
            self.raw_height
        } else {
            self.raw_width
        }
    }

    /// Effective display height after rotation.
    pub fn height(&self) -> u16 {
        if self.rotation.swaps() {
            self.raw_width
        } else {
            self.raw_height
        }
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
            0xD0, 0x04, 0x0D, 0x11, 0x13, 0x2B, 0x3F, 0x54, 0x4C, 0x18, 0x0D, 0x0B, 0x1F, 0x23,
        ])?;

        self.command(GMCTRN1)?;
        self.data(&[
            0xD0, 0x04, 0x0C, 0x11, 0x13, 0x2C, 0x3F, 0x44, 0x51, 0x2F, 0x1F, 0x1F, 0x20, 0x23,
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
