use std::{
    fs::{File, OpenOptions, create_dir_all},
    io::Write,
    path::PathBuf,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Level::Debug => write!(f, "DEBUG"),
            Level::Info => write!(f, "INFO"),
            Level::Warn => write!(f, "WARN"),
            Level::Error => write!(f, "ERROR"),
            Level::Fatal => write!(f, "FATAL"),
        }
    }
}

pub trait Logger: Send + Sync {
    fn log(&self, level: Level, file: &str, line: usize, message: &str);
}

pub static LOGGER: Mutex<Option<Box<dyn Logger>>> = Mutex::new(None);

pub fn format_timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let (year, month, day) = civil_from_days(days as i64);
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        year, month, day, hours, minutes, seconds
    )
}

pub fn format_today() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let days = secs / 86400;
    let (year, month, day) = civil_from_days(days as i64);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub struct StdoutLogger;

impl Logger for StdoutLogger {
    fn log(&self, level: Level, file: &str, line: usize, message: &str) {
        let timestamp = format_timestamp();
        let thread_id = std::thread::current().id();
        println!("[{:?}:{}:{} - {}:{}] {}", thread_id, level, timestamp, file, line, message);
    }
}

pub fn init_stdout_logger() {
    LOGGER.lock().unwrap().replace(Box::new(StdoutLogger));
}

struct FileLoggerState {
    path: PathBuf,
    current_date: String,
    file: File,
}

pub struct FileLogger {
    state: Mutex<FileLoggerState>,
}

impl FileLogger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        create_dir_all(&path).unwrap();
        let current_date = format_today();
        let path = path.join(format!("{}.log", current_date));
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        Self {
            state: Mutex::new(FileLoggerState {
                path,
                current_date,
                file,
            }),
        }
    }
}

impl Logger for FileLogger {
    fn log(&self, level: Level, file: &str, line: usize, message: &str) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let today = format_today();
        if today != state.current_date {
            let new_path = state.path.join(format!("{}.log", today));
            match OpenOptions::new().create(true).append(true).open(&new_path) {
                Ok(new_file) => {
                    state.file = new_file;
                    state.current_date = today;
                }
                Err(error) => {
                    eprintln!("Failed to open new log file {:?}: {}", new_path, error);
                }
            }
        }
        let timestamp = format_timestamp();
        let thread_id = std::thread::current().id();
        let log_line = format!("[{:?}:{}:{} - {}:{}] {}", thread_id, level, timestamp, file, line, message);
        if let Err(error) = state.file.write_all(log_line.as_bytes()) {
            eprintln!("Failed to write to log file: {}", error);
            eprintln!("{}", log_line.trim_end());
        }
    }
}

pub fn init_file_logger(path: impl Into<PathBuf>) {
    LOGGER.lock().unwrap().replace(Box::new(FileLogger::new(path)));
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{ let message = format_args!($($arg)*).to_string(); if let Some(logger) = base::log::LOGGER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { logger.log(base::log::Level::Debug, file!(), line!() as usize, &message); } }};
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{ let message = format_args!($($arg)*).to_string(); if let Some(logger) = base::log::LOGGER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { logger.log(base::log::Level::Info, file!(), line!() as usize, &message); } }};
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{ let message = format_args!($($arg)*).to_string(); if let Some(logger) = base::log::LOGGER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { logger.log(base::log::Level::Warn, file!(), line!() as usize, &message); } }};
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{ let message = format_args!($($arg)*).to_string(); if let Some(logger) = base::log::LOGGER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { logger.log(base::log::Level::Error, file!(), line!() as usize, &message); } }};
}

#[macro_export]
macro_rules! log_fatal {
    ($($arg:tt)*) => {{ let message = format_args!($($arg)*).to_string(); if let Some(logger) = base::log::LOGGER.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { logger.log(base::log::Level::Fatal, file!(), line!() as usize, &message); } println!("FATAL ERROR: {}", message); std::process::exit(1); }};
}
