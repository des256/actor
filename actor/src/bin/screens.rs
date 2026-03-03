use actor::{GpioOutput, Spi, SpiMode, rgb565};
use std::io;
use std::thread;
use std::time::Duration;

// Physical pin → BCM GPIO mapping
const GPIO_CHIP: u32 = 0; // gpiochip0 (use 4 on Pi 5)
const DC: u32 = 25;       // physical 22
const BL: u32 = 18;       // physical 12

const WIDTH: u16 = 240;
const HEIGHT: u16 = 240;

fn main() -> io::Result<()> {
    let dc = GpioOutput::new(GPIO_CHIP, DC)?;
    let bl = GpioOutput::new(GPIO_CHIP, BL)?;

    bl.set(false)?;
    thread::sleep(Duration::from_millis(100));
    bl.set(true)?;

    let left = Spi::open(0, 0)?;   // CE0, physical 24
    let right = Spi::open(0, 1)?;  // CE1, physical 26
    for spi in [&left, &right] {
        spi.set_mode(SpiMode::Mode0)?;
        spi.set_lsb_first(false)?;
        spi.set_max_speed_hz(4_000_000)?;
    }

    init_display(&left, &dc)?;
    init_display(&right, &dc)?;

    let red = rgb565(255, 0, 0);
    fill(&left, &dc, red)?;
    fill(&right, &dc, red)?;

    println!("both screens red — ctrl-c to exit");
    loop {
        thread::sleep(Duration::from_secs(60));
    }
}

// -- low-level helpers (mirror st7789.rs init sequence) ----------------------

fn cmd(spi: &Spi, dc: &GpioOutput, c: u8) -> io::Result<()> {
    dc.set(false)?;
    spi.write(&[c])
}

fn dat(spi: &Spi, dc: &GpioOutput, d: &[u8]) -> io::Result<()> {
    dc.set(true)?;
    for chunk in d.chunks(4096) {
        spi.write(chunk)?;
    }
    Ok(())
}

fn init_display(spi: &Spi, dc: &GpioOutput) -> io::Result<()> {
    cmd(spi, dc, 0x01)?;                                       // SWRESET
    thread::sleep(Duration::from_millis(150));

    cmd(spi, dc, 0x36)?; dat(spi, dc, &[0x00])?;               // MADCTL
    cmd(spi, dc, 0xB2)?; dat(spi, dc, &[0x0C,0x0C,0x00,0x33,0x33])?; // FRMCTR2
    cmd(spi, dc, 0x3A)?; dat(spi, dc, &[0x05])?;               // COLMOD 16-bit
    cmd(spi, dc, 0xB7)?; dat(spi, dc, &[0x14])?;               // GCTRL
    cmd(spi, dc, 0xBB)?; dat(spi, dc, &[0x37])?;               // VCOMS
    cmd(spi, dc, 0xC0)?; dat(spi, dc, &[0x2C])?;               // LCMCTRL
    cmd(spi, dc, 0xC2)?; dat(spi, dc, &[0x01])?;               // VDVVRHEN
    cmd(spi, dc, 0xC3)?; dat(spi, dc, &[0x12])?;               // VRHS
    cmd(spi, dc, 0xC4)?; dat(spi, dc, &[0x20])?;               // VDVS
    cmd(spi, dc, 0xD0)?; dat(spi, dc, &[0xA4, 0xA1])?;         // PWCTRL1
    cmd(spi, dc, 0xC6)?; dat(spi, dc, &[0x0F])?;               // FRCTRL2

    cmd(spi, dc, 0xE0)?;                                        // GMCTRP1
    dat(spi, dc, &[
        0xD0,0x04,0x0D,0x11,0x13,0x2B,0x3F,0x54,
        0x4C,0x18,0x0D,0x0B,0x1F,0x23,
    ])?;
    cmd(spi, dc, 0xE1)?;                                        // GMCTRN1
    dat(spi, dc, &[
        0xD0,0x04,0x0C,0x11,0x13,0x2C,0x3F,0x44,
        0x51,0x2F,0x1F,0x1F,0x20,0x23,
    ])?;

    cmd(spi, dc, 0x21)?;                                        // INVON
    cmd(spi, dc, 0x11)?;                                        // SLPOUT
    cmd(spi, dc, 0x29)?;                                        // DISPON
    thread::sleep(Duration::from_millis(100));
    Ok(())
}

fn fill(spi: &Spi, dc: &GpioOutput, color: [u8; 2]) -> io::Result<()> {
    let w = WIDTH;
    let h = HEIGHT;

    // set_window(0, 0, w-1, h-1)
    cmd(spi, dc, 0x2A)?;
    dat(spi, dc, &[0, 0, ((w - 1) >> 8) as u8, (w - 1) as u8])?;
    cmd(spi, dc, 0x2B)?;
    dat(spi, dc, &[0, 0, ((h - 1) >> 8) as u8, (h - 1) as u8])?;
    cmd(spi, dc, 0x2C)?; // RAMWR

    // stream the color
    let mut buf = [0u8; 4096];
    for i in (0..buf.len()).step_by(2) {
        buf[i] = color[0];
        buf[i + 1] = color[1];
    }
    let total = w as usize * h as usize * 2;
    dc.set(true)?;
    let mut sent = 0;
    while sent < total {
        let n = (total - sent).min(buf.len());
        spi.write(&buf[..n])?;
        sent += n;
    }
    Ok(())
}
