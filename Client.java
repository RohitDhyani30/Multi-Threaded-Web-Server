package pool;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Client {

    // Define task (Runnable) for each client connection
    public Runnable getRunnable() {
        return new Runnable() {
            @Override
            public void run() {
                int port = 8010;
                try {
                    InetAddress address = InetAddress.getByName("localhost");
                    try (Socket socket = new Socket(address, port);
                         PrintWriter toSocket = new PrintWriter(socket.getOutputStream(), true);
                         BufferedReader fromSocket = new BufferedReader(new InputStreamReader(socket.getInputStream()))
                    ) {
                        // Send message to server
                        toSocket.println("Hello from Client " + socket.getLocalSocketAddress());

                        // Receive and print response from server
                        String response = fromSocket.readLine();
                        System.out.println("Response from Server: " + response);

                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        };
    }

    public static void main(String[] args) {
        Client client = new Client();

        // Create a thread pool with 50 threads
        ExecutorService executor = Executors.newFixedThreadPool(50);

        // Run 100 client requests using the same pool
        for (int i = 0; i < 100; i++) {
            executor.submit(client.getRunnable());
        }

        // Shut down the client thread pool after all tasks are submitted
        executor.shutdown();
        System.out.println("All client requests submitted to thread pool.");
    }
}
