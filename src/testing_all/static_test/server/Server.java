package testing_all.static_test.server;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server {

    private static final int THREAD_POOL_SIZE = 50; // Your static size
    private static ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_POOL_SIZE);

    public SocketHandler getConsumer() {
        return new SocketHandler();
    }

    public static void main(String[] args) {
        int port = 8010;
        Server server = new Server();

        // Add a shutdown hook to cleanly stop the thread pool
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutting down thread pool...");
            threadPool.shutdown();
        }));

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("Server is listening on port " + port);
            System.out.println("Static thread pool size: " + THREAD_POOL_SIZE);

            while (true) {
                Socket clientSocket = serverSocket.accept();
                SocketHandler handler = server.getConsumer();
                threadPool.submit(new ClientHandlerRunnable(clientSocket, handler));
            }
        } catch (IOException ex) {
            if (!threadPool.isShutdown()) {
                ex.printStackTrace();
            }
        }
    }
}