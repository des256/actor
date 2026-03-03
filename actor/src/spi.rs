use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;

const SPI_IOC_MAGIC: u8 = b'k';

const fn ioc(dir: u8, ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ((dir as libc::c_ulong) << 30)
        | ((ty as libc::c_ulong) << 8)
        | (nr as libc::c_ulong)
        | ((size as libc::c_ulong) << 16)
}

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
    tx_nbits: u32,
    rx_nbits: u32,
    word_delay_usecs: u16,
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
                let rx_ptr = t
                    .rx
                    .as_deref_mut()
                    .map_or(std::ptr::null_mut(), |b| b.as_mut_ptr());
                let len = t.tx.map_or(0, |b| b.len()).max(
                    t.rx.as_deref()
                        .map_or(0, |b| b.len()),
                );
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
