package pool;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public class Server {

    // Define a Consumer that handles each client connection
    public Consumer<Socket> getConsumer() {
        return (clientSocket) -> {
            System.out.println("Accepted connection from: " + clientSocket.getRemoteSocketAddress());
            try (PrintWriter toSocket = new PrintWriter(clientSocket.getOutputStream(), true)) {
                // Send response to the client
                toSocket.println("Hello from Server " + clientSocket.getInetAddress());
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    clientSocket.close(); // Always close client socket after handling
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        };
    }

    public static void main(String[] args) {
        int port = 8010;
        Server server = new Server();

        // Create a Thread Pool with 50 worker threads
        ExecutorService executor = Executors.newFixedThreadPool(50);

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            serverSocket.setSoTimeout(70000);
            System.out.println("Server is listening on port " + port);

            // Infinite loop to accept multiple client connections
            while (true) {
                Socket clientSocket = serverSocket.accept();

                // Submit each client handling task to the thread pool
                executor.submit(() -> server.getConsumer().accept(clientSocket));
            }

        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            // Gracefully shut down the thread pool when the server stops
            executor.shutdown();
            System.out.println("Server stopped, thread pool shut down.");
        }
    }
}
