use {
    crate::*,
    futures_util::{SinkExt, StreamExt, stream::SplitSink},
    std::{collections::HashMap, net::SocketAddr, sync::Arc},
    tokio::{
        net::{TcpListener, TcpStream, ToSocketAddrs},
        sync::{RwLock, mpsc as tokio_mpsc},
    },
    tokio_websockets::{Message, ServerBuilder, WebSocketStream},
};

const CHANNEL_CAPACITY: usize = 1024;

pub struct WsServerHandle {
    sinks: Arc<RwLock<HashMap<SocketAddr, SplitSink<WebSocketStream<TcpStream>, Message>>>>,
}

pub struct WsServerListener<T> {
    rx: tokio_mpsc::Receiver<T>,
}

pub async fn create_wsserver<T: Codec + Send + 'static>(
    addr: impl ToSocketAddrs,
) -> (WsServerHandle, WsServerListener<T>) {
    let listener = TcpListener::bind(addr).await.unwrap();
    let sinks: Arc<RwLock<HashMap<SocketAddr, SplitSink<WebSocketStream<TcpStream>, Message>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let (tx, rx) = tokio_mpsc::channel::<T>(CHANNEL_CAPACITY);
    tokio::spawn({
        let sinks = Arc::clone(&sinks);
        async move {
            while let Ok((stream, addr)) = listener.accept().await {
                let stream = match ServerBuilder::new().accept(stream).await {
                    Ok((_, stream)) => stream,
                    Err(error) => {
                        println!("WsServer: failed to accept connection: {}", error);
                        continue;
                    }
                };
                let (sink, mut source) = stream.split();
                sinks.write().await.insert(addr, sink);
                tokio::spawn({
                    let sinks = Arc::clone(&sinks);
                    let tx = tx.clone();
                    async move {
                        while let Some(Ok(message)) = source.next().await {
                            if message.is_binary() {
                                let payload = message.into_payload();
                                let value = T::decode(&payload);
                                if let Err(error) = tx.send(value).await {
                                    println!("WsServer: failed to send message: {}", error);
                                    sinks.write().await.remove(&addr);
                                }
                            }
                        }
                        sinks.write().await.remove(&addr);
                    }
                });
            }
        }
    });
    (
        WsServerHandle {
            sinks: Arc::clone(&sinks),
        },
        WsServerListener { rx },
    )
}

impl WsServerHandle {
    pub async fn broadcast<T: Codec>(&self, value: T) {
        let size = value.size();
        let mut payload = Vec::<u8>::with_capacity(size);
        value.encode(&mut payload);
        let message = Message::binary(payload);
        let mut failed_sinks = Vec::new();
        let mut write = self.sinks.write().await;
        for (addr, sink) in write.iter_mut() {
            if let Err(_) = sink.send(message.clone()).await {
                failed_sinks.push(*addr);
            }
        }
        for addr in failed_sinks {
            write.remove(&addr);
        }
    }
}

impl<T: Codec + Send + 'static> WsServerListener<T> {
    pub async fn recv(&mut self) -> Option<T> {
        self.rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<T> {
        match self.rx.try_recv() {
            Ok(value) => Some(value),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("WsServer: data channel disconnected");
            }
        }
    }
}
