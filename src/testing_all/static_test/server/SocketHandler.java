package testing_all.static_test.server;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.function.Consumer;

public class SocketHandler implements Consumer<Socket> {

    @Override
    public void accept(Socket clientSocket) {
        // Note: This will print a LOT. For high-load tests, you might want
        // to comment out the System.out.println line.
        System.out.println("Accepted connection from: " + clientSocket.getRemoteSocketAddress());

        try (PrintWriter toSocket = new PrintWriter(clientSocket.getOutputStream(), true)) {
            // Send a simple reply
            toSocket.println("Hello from server " + clientSocket.getInetAddress());
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                clientSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}