package testing_all.static_test.client;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class Client {
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final List<Future<?>> clients = new ArrayList<>();
    private final PrintWriter csvWriter;

    public Client() {
        try {
            csvWriter = new PrintWriter(new FileWriter("C:/Users/mohit/OneDrive/Desktop/multithreaded/src/testing_all/client_log_static.csv", false));
            csvWriter.println("Timestamp,Phase,LatencyMs,Error");
            csvWriter.flush();
        } catch (IOException e) {
            throw new RuntimeException("Failed to init client log", e);
        }
    }

    public Runnable createTask(int id, int delayMs, String phaseName) {
        return () -> {
            while (!Thread.currentThread().isInterrupted()) {
                long startTime = 0;
                long latency = -1;
                int error = 0;

                try {
                    startTime = System.nanoTime();
                    try (Socket s = new Socket("localhost", 8010);
                         PrintWriter out = new PrintWriter(s.getOutputStream(), true);
                         BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()))) {

                        s.setSoTimeout(2000); // 2-second timeout
                        out.println("REQ-" + id); // Send a request
                        in.readLine(); // Wait for the server's "Hello..." reply
                        latency = (System.nanoTime() - startTime) / 1_000_000;
                    }
                } catch (IOException e) {
                    error = 1; // Mark as an error (e.g., timeout, connection refused)
                    latency = (System.nanoTime() - startTime) / 1_000_000;
                }

                // Log the result
                logToCsv(phaseName, latency, error);

                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        };
    }

    private synchronized void logToCsv(String phase, long latency, int error) {
        if (csvWriter != null) {
            csvWriter.printf("%d,%s,%d,%d%n",
                    System.currentTimeMillis(),
                    phase,
                    latency,
                    error
            );
            csvWriter.flush(); // Flush regularly so we can see data live
        }
    }

    public void setClients(int targetCount, int delayMs, String phaseName) {
        int current = clients.size();

        if (targetCount > current) {
            int toAdd = targetCount - current;
            System.out.printf("[%s] Adding %d clients (delay=%dms) → Total: %d%n",
                    phaseName, toAdd, delayMs, targetCount);
            for (int i = 0; i < toAdd; i++) {
                clients.add(executor.submit(createTask(clients.size() + 1, delayMs, phaseName)));
            }
        } else if (targetCount < current) {
            int toRemove = current - targetCount;
            System.out.printf("[%s] Removing %d clients → Total: %d%n",
                    phaseName, toRemove, targetCount);
            for (int i = 0; i < toRemove; i++) {
                clients.remove(clients.size() - 1).cancel(true);
            }
        } else {
            System.out.printf("[%s] Maintaining %d clients (delay=%dms)%n",
                    phaseName, targetCount, delayMs);
        }
    }

    public void stopAll() {
        executor.shutdownNow();
        if (csvWriter != null) {
            csvWriter.close();
            System.out.println("\nClient log saved to client_log.csv");
        }
        clients.clear();
    }

    public static void main(String[] args) throws InterruptedException {
        Client client = new Client();
        Runtime.getRuntime().addShutdownHook(new Thread(client::stopAll));

        System.out.println("--- LOAD GENERATOR STARTED ---");
        System.out.println("Logging latency and errors to client_log.csv\n");

        int cycle = 1;
        while (true) {
            System.out.printf("%n━━━━━━━━━━━━━━━━ CYCLE %d ━━━━━━━━━━━━━━━━%n", cycle++);

            client.setClients(3, 200, "LOW");
            Thread.sleep(35000);

            client.setClients(6, 100, "NORMAL");
            Thread.sleep(35000);

            client.setClients(16, 80, "HIGH");
            Thread.sleep(35000);

            client.setClients(30, 60, "EXTREME");
            Thread.sleep(35000);

            System.out.println("\n[SCALE DOWN INITIATED]");
            client.setClients(16, 80, "HIGH-RETURN");
            Thread.sleep(30000);

            client.setClients(6, 100, "NORMAL-RETURN");
            Thread.sleep(30000);

            client.setClients(3, 200, "LOW-RETURN");
            Thread.sleep(30000);

            client.setClients(0, 0, "RESET");
            Thread.sleep(20000);
        }
    }
}