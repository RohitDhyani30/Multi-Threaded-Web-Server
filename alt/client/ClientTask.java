package alt.client;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;

// Named Runnable class for a single client task
public class ClientTask implements Runnable {

    private String host;
    private int port;

    public ClientTask(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public void run() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                InetAddress address = InetAddress.getByName(host);
                try (Socket socket = new Socket(address, port);
                     PrintWriter toSocket = new PrintWriter(socket.getOutputStream(), true);
                     BufferedReader fromSocket = new BufferedReader(new InputStreamReader(socket.getInputStream()))
                ) {
                    toSocket.println("Hello from Client " + socket.getLocalSocketAddress());
                    String line = fromSocket.readLine();
                    System.out.println("Response from Server: " + line);
                } catch (IOException e) {
                    e.printStackTrace();
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
}
