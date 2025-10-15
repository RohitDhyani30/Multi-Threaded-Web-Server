package alt.server;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.function.Consumer;

// Named Consumer class
public class SocketHandler implements Consumer<Socket> {

    @Override
    public void accept(Socket clientSocket) {
        System.out.println("Accepted connection from: " + clientSocket.getRemoteSocketAddress());
        PrintWriter toSocket = null;
        try {
            toSocket = new PrintWriter(clientSocket.getOutputStream(), true);
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