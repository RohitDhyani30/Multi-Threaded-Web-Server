package alt.client;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Client {

    private static final int THREAD_POOL_SIZE = 50;
    private ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_POOL_SIZE);
    private String host;
    private int port;

    public Client(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void startClients(int numClients) {
        for (int i = 0; i < numClients; i++) {
            threadPool.submit(new ClientTask(host, port));
        }
    }

    public static void main(String[] args) {
        String host = "localhost";
        int port = 8010;
        int numClients = 100;

        Client client = new Client(host, port);
        System.out.println("Starting " + numClients + " continuous clients with thread pool size: " + THREAD_POOL_SIZE);
        client.startClients(numClients);

        // Do not shut down the thread pool to keep clients running
    }
}
