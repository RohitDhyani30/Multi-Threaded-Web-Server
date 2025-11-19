package dynamic;

import java.io.PrintWriter;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Client {
    private final ExecutorService threadPool = Executors.newCachedThreadPool();
    private final List<Future<?>> activeTasks = new ArrayList<>();

    public void updateLoad(int targetCount, int delayMs, String phase) {
        System.out.printf("[%s]", phase);

        while (activeTasks.size() < targetCount) {
            int id = activeTasks.size() + 1;
            activeTasks.add(threadPool.submit(() -> runTask(id, delayMs)));
        }

        while (activeTasks.size() > targetCount) {
            activeTasks.remove(activeTasks.size() - 1).cancel(true);
        }
    }

    private void runTask(int id, int delayMs) {
        while (!Thread.currentThread().isInterrupted()) {
            try (Socket socket = new Socket("localhost", 8010);
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                 Scanner in = new Scanner(socket.getInputStream())) {

                socket.setSoTimeout(2000);
                out.println("REQ-" + id);
                if (in.hasNext()) in.nextLine();

                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception ignored) {}
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Client client = new Client();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> client.threadPool.shutdownNow()));

        int cycle = 1;
        while (true) {
            System.out.printf("%n--- CYCLE %d ---%n", cycle++);

            client.updateLoad(3, 200, "LOW");
            Thread.sleep(35000);

            client.updateLoad(6, 100, "NORMAL");
            Thread.sleep(35000);

            client.updateLoad(16, 80, "HIGH");
            Thread.sleep(35000);

            client.updateLoad(30, 60, "EXTREME");
            Thread.sleep(35000);

            System.out.println("\n[SCALE DOWN]");
            client.updateLoad(16, 80, "HIGH-RETURN");
            Thread.sleep(30000);

            client.updateLoad(6, 100, "NORMAL-RETURN");
            Thread.sleep(30000);

            client.updateLoad(3, 200, "LOW-RETURN");
            Thread.sleep(30000);

            client.updateLoad(0, 0, "RESET");
            Thread.sleep(20000);
        }
    }
}