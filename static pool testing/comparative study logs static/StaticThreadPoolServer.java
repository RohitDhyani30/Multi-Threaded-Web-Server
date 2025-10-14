import java.io.*;
import java.net.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class StaticThreadPoolServer {
    private static final int PORT = 8010, POOL_SIZE = 50, PHASES = 3;
    private static final AtomicInteger totalReq = new AtomicInteger(), completeReq = new AtomicInteger();
    private static final AtomicInteger[] phaseSamples = new AtomicInteger[PHASES];
    private static final AtomicLong[] idleThreadSum = new AtomicLong[PHASES];
    private static final AtomicInteger[] saturatedSamples = new AtomicInteger[PHASES];
    private static final AtomicReference<LoadPhase> currentPhase = new AtomicReference<>(LoadPhase.NORMAL);
    private static final AtomicInteger reqPer5s = new AtomicInteger();
    private static volatile boolean running = true;
    private static BufferedWriter logWriter;
    private static final String LOG_FILE = "static_server_log.csv";

    static {
        for (int i = 0; i < PHASES; i++) {
            phaseSamples[i] = new AtomicInteger();
            idleThreadSum[i] = new AtomicLong();
            saturatedSamples[i] = new AtomicInteger();
        }
        try {
            logWriter = new BufferedWriter(new FileWriter(LOG_FILE, false));
            logWriter.write("Timestamp,PoolSize,ActiveThreads,Utilization,ReqPer5s\n");
        } catch (IOException e) {
            System.err.println("Failed to initialize log file: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.printf("\n==== StaticThreadPoolServer STARTED ====\nListening: Port %d | Thread pool: %d\n", PORT, POOL_SIZE);
        ExecutorService executor = Executors.newFixedThreadPool(POOL_SIZE);
        ServerSocket serverSocket = new ServerSocket(PORT);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            running = false;
            try { serverSocket.close(); } catch (Exception ignored) {}
            executor.shutdownNow();
            printPerformanceSummary();
            try { logWriter.close(); } catch (IOException ignored) {}
        }));
        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(
            () -> samplePhaseStatus(executor), 1, 1, TimeUnit.SECONDS);
        try {
            while (running) {
                Socket clientSocket = serverSocket.accept();
                totalReq.incrementAndGet();
                executor.submit(() -> handleClient(clientSocket));
            }
        } catch (IOException e) {
            if (running) System.out.println(" [Server] Accept error: " + e.getMessage());
        }
    }

    private static void handleClient(Socket clientSocket) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
             PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
            String request = in.readLine();
            LoadPhase phase = LoadPhase.NORMAL;
            if (request != null) {
                if (request.contains("PHASE=LOW")) phase = LoadPhase.LOW;
                else if (request.contains("PHASE=HIGH")) phase = LoadPhase.HIGH;
            }
            currentPhase.set(phase);
            reqPer5s.incrementAndGet();
            int delay = switch (phase) {
                case HIGH -> 120 + (int)(Math.random() * 70);
                case LOW -> 30 + (int)(Math.random() * 25);
                default -> 60 + (int)(Math.random() * 25);
            };
            Thread.sleep(delay);
            out.println("Server ACK for " + request + " at " + System.currentTimeMillis());
            System.out.printf(" [Server] %s handled (Thread: %s)\n", request, Thread.currentThread().getName());
        } catch (Exception e) {
            System.out.println(" [Server] Error: " + e.getMessage());
        } finally {
            completeReq.incrementAndGet();
            try { clientSocket.close(); } catch (IOException ignored) {}
        }
    }

    private static void samplePhaseStatus(ExecutorService executor) {
        int active = ((ThreadPoolExecutor) executor).getActiveCount();
        int poolSize = POOL_SIZE;
        int idle = poolSize - active;
        int reqCount = reqPer5s.getAndSet(0);
        int idx = currentPhase.get().ordinal();
        phaseSamples[idx].incrementAndGet();
        idleThreadSum[idx].addAndGet(idle);
        if (active == poolSize) saturatedSamples[idx].incrementAndGet();

        double utilization = poolSize == 0 ? 0.0 : (active * 100.0 / poolSize);
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        try {
            logWriter.write(String.format("%s,%d,%d,%.2f,%d\n",
                timestamp, poolSize, active, utilization, reqCount));
            logWriter.flush();
        } catch (IOException e) {
            System.err.println("Log write error: " + e.getMessage());
        }
    }

    private static void printPerformanceSummary() {
        System.out.println("\n=== Server Session Thread Pool Performance ===");
        String[] phNames = { "LOW   ", "NORMAL", "HIGH  " };
        for (LoadPhase phase : LoadPhase.values()) {
            int idx = phase.ordinal(), samples = phaseSamples[idx].get();
            long idleAvg = samples == 0 ? 0 : idleThreadSum[idx].get() / samples;
            int saturated = saturatedSamples[idx].get();
            double saturationPct = samples == 0 ? 0.0 : saturated * 100.0 / samples;
            System.out.printf("- %s phase: Avg idle threads: %2d, Saturated: %d times (%.1f%% of phase)\n",
                              phNames[idx], idleAvg, saturated, saturationPct);
        }
        System.out.printf("Total requests: %d | Completed: %d\n", totalReq.get(), completeReq.get());
        System.out.println("Dynamic pool sizing can reduce idle threads in LOW, and avoid saturation in HIGH.");
    }
}