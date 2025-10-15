package DatasetGeneration;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class LoadSimulatorClient {

    private static final String HOST = "127.0.0.1";
    private static final int PORT = 8010, MAX_THREADS = 50;
    private static final String MESSAGE = "Client request";

    private static record Phase(String name, long delayMs, int durationSec) {}

    private static final List<Phase> PHASES = List.of(
            new Phase("Low Load", 300, 30),
            new Phase("Medium Load", 100, 60),
            new Phase("High Load", 20, 60),
            new Phase("Cool Down", 400, 30)
    );

    private static class ClientTask implements Runnable {
        private final String phase;
        ClientTask(String phase) { this.phase = phase; }

        public void run() {
            long start = System.nanoTime();
            try (Socket socket = new Socket(HOST, PORT);
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                out.println(MESSAGE);
                String response = in.readLine();
                long latency = (System.nanoTime() - start) / 1_000_000;
                System.out.printf("[%s] Latency=%dms | Response=%s%n", phase, latency, response);
            } catch (IOException e) {
                System.err.printf("[%s] Error: %s%n", phase, e.getMessage());
            }
        }
    }

    private static void simulatePhase(ExecutorService pool, Phase phase) {
        System.out.printf("%n=== Phase: %s (delay %dms, duration %ds) ===%n", phase.name(), phase.delayMs(), phase.durationSec());
        long end = System.currentTimeMillis() + phase.durationSec() * 1000L;
        while (System.currentTimeMillis() < end) {
            pool.submit(new ClientTask(phase.name()));
            try { Thread.sleep(phase.delayMs()); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }
        System.out.printf("=== Phase %s ended ===%n", phase.name());
    }

    public static void main(String[] args) {
        System.out.println("Load Simulator Client started...");
        ExecutorService pool = Executors.newFixedThreadPool(MAX_THREADS);

        while (true) {
            List<Phase> shuffledPhases = new ArrayList<>(PHASES);
            Collections.shuffle(shuffledPhases); // randomize phase order

            for (Phase phase : shuffledPhases) {
                simulatePhase(pool, phase);
            }

            try {
                Thread.sleep(5000); // brief pause between cycles
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        pool.shutdown();
        try {
            pool.awaitTermination(60, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("Client stopped.");
    }
}