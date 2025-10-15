package alt.server;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server {

    private static final int THREAD_POOL_SIZE = 50;
    private static ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_POOL_SIZE);

    // Return an instance of named Consumer class
    public SocketHandler getConsumer() {
        return new SocketHandler();
    }

    public static void main(String[] args) {
        int port = 8010;
        Server server = new Server();

        try {
            ServerSocket serverSocket = new ServerSocket(port);
            serverSocket.setSoTimeout(70000);
            System.out.println("Server is listening on port " + port);
            System.out.println("Thread pool size: " + THREAD_POOL_SIZE);

            while (true) {
                Socket clientSocket = serverSocket.accept();
                SocketHandler handler = server.getConsumer();

                // Submit a named Runnable instead of anonymous class
                threadPool.submit(new ClientHandlerRunnable(clientSocket, handler));
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            server.threadPool.shutdown();
        }
    }
}
