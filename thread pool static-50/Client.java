import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Client {
    private static final int THREAD_POOL_SIZE = 50;
    private ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_POOL_SIZE);

    public Runnable getRunnable() {
        return new Runnable() {
            @Override
            public void run() {
                int port = 8010;
                while (!Thread.currentThread().isInterrupted()) {
                    try {
                        InetAddress address = InetAddress.getByName("localhost");
                        Socket socket = new Socket(address, port);
                        try (
                            PrintWriter toSocket = new PrintWriter(socket.getOutputStream(), true);
                            BufferedReader fromSocket = new BufferedReader(new InputStreamReader(socket.getInputStream()))
                        ) {
                            toSocket.println("Hello from Client " + socket.getLocalSocketAddress());
                            String line = fromSocket.readLine();
                            System.out.println("Response from Server: " + line);
                        } catch (IOException e) {
                            e.printStackTrace();
                        } finally {
                            socket.close();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    try {
                        Thread.sleep(200); // brief pause between requests
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        };
    }

    public static void main(String[] args) {
        Client client = new Client();
        System.out.println("Starting 100 continuous clients with thread pool size: " + THREAD_POOL_SIZE);

        for (int i = 0; i < 100; i++) {
            client.threadPool.submit(client.getRunnable());
        }

        // Do not shut down the thread pool to keep clients running
    }
}
