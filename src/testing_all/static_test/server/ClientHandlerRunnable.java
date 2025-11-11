package testing_all.static_test.server;
import java.net.Socket;

public class ClientHandlerRunnable implements Runnable {

    private Socket clientSocket;
    private SocketHandler socketHandler;

    public ClientHandlerRunnable(Socket clientSocket, SocketHandler socketHandler) {
        this.clientSocket = clientSocket;
        this.socketHandler = socketHandler;
    }

    @Override
    public void run() {
        socketHandler.accept(clientSocket);
    }
}