package demo;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public class Server {
    private static final int THREAD_POOL_SIZE = 50;
    private ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_POOL_SIZE);

    public Consumer<Socket> getConsumer() {
        return new Consumer<Socket>() {
            @Override
            public void accept(Socket clientSocket) {
                System.out.println("Accepted connection from: " + clientSocket.getRemoteSocketAddress());
                PrintWriter toSocket = null;
                try
                {
                    toSocket = new PrintWriter(clientSocket.getOutputStream(), true);
                    toSocket.println("Hello from server " + clientSocket.getInetAddress());
                }
                catch (IOException ex)
                {
                    ex.printStackTrace();
                }
                finally
                {
                    try
                    {
                        clientSocket.close();
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                    }
                }
            }
        };
    }

    public static void main(String[] args) {
        int port = 8010;
        final Server server = new Server(); // final to allow inner class access

        try
        {
            final ServerSocket serverSocket = new ServerSocket(port);
            serverSocket.setSoTimeout(70000);
            System.out.println("Server is listening on port " + port);
            System.out.println("Thread pool size: " + THREAD_POOL_SIZE);

            while (true) {
                final Socket clientSocket = serverSocket.accept();
                // Submit Runnable to thread pool instead of lambda
                server.threadPool.submit(new Runnable() {
                    @Override
                    public void run() {
                        server.getConsumer().accept(clientSocket);
                    }
                });
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            server.threadPool.shutdown();
        }
    }
}
