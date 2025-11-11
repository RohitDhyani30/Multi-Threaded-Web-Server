package dynamic;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class Client {
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final List<Future<?>> clients = new ArrayList<>();

    public Runnable createTask(int id, int delayMs) {
        return () -> {
            while (!Thread.currentThread().isInterrupted()) {
                try (Socket s = new Socket("localhost", 8010);
                     PrintWriter out = new PrintWriter(s.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()))) {
                    s.setSoTimeout(2000);
                    out.println("REQ-" + id);
                    in.readLine(); // Wait for server 'OK'
                } catch (IOException ignored) {}
                
                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        };
    }

    public void setClients(int targetCount, int delayMs, String phaseName) {
        int current = clients.size();
        
        if (targetCount > current) {
            int toAdd = targetCount - current;
            System.out.printf("[%s] Adding %d clients (delay=%dms) → Total: %d%n", 
                phaseName, toAdd, delayMs, targetCount);
            for (int i = 0; i < toAdd; i++) {
                clients.add(executor.submit(createTask(clients.size() + 1, delayMs)));
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
        clients.clear();
    }

    public static void main(String[] args) throws InterruptedException {
        Client client = new Client();
        Runtime.getRuntime().addShutdownHook(new Thread(client::stopAll));

        System.out.println("--- LOAD GENERATOR STARTED ---");
        System.out.println("Low:     3 clients @ 200ms");
        System.out.println("Normal:  6 clients @ 100ms");
        System.out.println("High:    16 clients @ 80ms");
        System.out.println("Extreme: 30 clients @ 60ms\n");

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