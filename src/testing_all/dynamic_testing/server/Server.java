package testing_all.dynamic_testing.server; // <-- NEW PACKAGE LINE

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

public class Server {
    private final int MIN = 10, MAX = 160;
    private ThreadPoolExecutor pool;
    private final AtomicLong totalReq = new AtomicLong();
    private PrintWriter csvWriter;

    public Server() {
        pool = new ThreadPoolExecutor(
                MIN, MAX,
                60, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(2000),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        pool.allowCoreThreadTimeOut(false);
        pool.prestartAllCoreThreads();

        try {
            // Log to the current directory
            csvWriter = new PrintWriter(new FileWriter("C:/Users/mohit/OneDrive/Desktop/multithreaded/src/testing_all/thread_analysis.csv", false));
            csvWriter.println("Timestamp,PoolSize,ActiveThreads,Utilization,QueueSize,ReqPer5s,Phase,MLSuggested,TargetCalculated");
            csvWriter.flush();
        } catch (IOException e) {
            System.err.println("Failed to initialize CSV logging");
        }
    }

    public Consumer<Socket> getHandler() {
        return client -> {
            try (PrintWriter pw = new PrintWriter(client.getOutputStream(), true)) {
                Thread.sleep(50); // Simulate work
                pw.println("OK-" + totalReq.incrementAndGet());
            } catch (Exception ignored) {
            } finally {
                try { client.close(); } catch (IOException ignored) {}
            }
        };
    }

    private void reportToML(long requestCount) {
        try {
            URL url = new URI("http://localhost:5000/ml/update_load/" + requestCount).toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setConnectTimeout(500);
            conn.setReadTimeout(500);
            conn.getResponseCode(); // Fire and forget
            conn.disconnect();
        } catch (Exception ignored) {}
    }

    public void startMonitor() {
        new Thread(() -> {
            long lastTotal = 0;
            while (true) {
                try {
                    Thread.sleep(5000);

                    long currentTotal = totalReq.get();
                    long requestDelta = currentTotal - lastTotal;
                    lastTotal = currentTotal;

                    reportToML(requestDelta);

                    int mlSuggested = 50;
                    String phase = "Unknown";

                    try {
                        URL url = new URI("http://localhost:5000/ml/suggest_threads").toURL();
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("GET");
                        conn.setConnectTimeout(1000);
                        conn.setReadTimeout(1000);

                        try (Scanner sc = new Scanner(conn.getInputStream())) {
                            String response = sc.useDelimiter("\\A").next();
                            mlSuggested = extractInt(response, "suggested_threads");
                            phase = extractString(response, "phase");
                        }
                        conn.disconnect();
                    } catch (Exception e) {
                        // Fallback: simple reactive scaling
                        mlSuggested = Math.max(MIN, Math.min(MAX, (int)(requestDelta / 5.0)));
                    }

                    int active = pool.getActiveCount();
                    int poolSize = pool.getPoolSize();
                    int currentCore = pool.getCorePoolSize();
                    int queueSize = pool.getQueue().size();

                    // Dynamic sizing: 70% reactive (active threads) + 30% proactive (ML)
                    int targetSize = (int)((active + 8) * 0.7 + mlSuggested * 0.3);

                    // Queue pressure handling
                    if (queueSize > 500) {
                        targetSize = Math.max(targetSize, currentCore + 30);
                    } else if (queueSize > 100) {
                        targetSize = Math.max(targetSize, currentCore + 15);
                    }

                    targetSize = Math.max(MIN, Math.min(MAX, targetSize)); // Enforce bounds

                    // Log to CSV
                    double utilization = poolSize > 0 ? (active * 100.0 / poolSize) : 0;
                    if (csvWriter != null) {
                        csvWriter.printf("%d,%d,%d,%.2f,%s,%d,%d%n",
                                System.currentTimeMillis(), poolSize, active, utilization,
                                phase, mlSuggested, targetSize);
                        csvWriter.flush();
                    }

                    // Resize if needed
                    if (Math.abs(targetSize - currentCore) >= 3) {
                        if (targetSize > currentCore) {
                            pool.setMaximumPoolSize(targetSize);
                            pool.setCorePoolSize(targetSize);
                        } else {
                            pool.setCorePoolSize(targetSize);
                            pool.setMaximumPoolSize(targetSize);
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "PoolMonitor").start();
    }

    private int extractInt(String json, String key) {
        try {
            java.util.regex.Matcher m = java.util.regex.Pattern
                    .compile("\"" + key + "\":\\s*(\\d+)").matcher(json);
            if (m.find()) return Integer.parseInt(m.group(1));
        } catch (Exception e) {}
        return 0;
    }

    private String extractString(String json, String key) {
        try {
            java.util.regex.Matcher m = java.util.regex.Pattern
                    .compile("\"" + key + "\":\\s*\"([^\"]+)\"").matcher(json);
            if (m.find()) return m.group(1);
        } catch (Exception e) {}
        return "Unknown";
    }

    public static void main(String[] args) {
        Server server = new Server();
        server.startMonitor();

        System.out.println("✅ Dynamic Server starting on 8010...");
        System.out.println("📊 Logging metrics to thread_analysis.csv\n");

        try (ServerSocket serverSocket = new ServerSocket(8010, 500)) {
            while (true) {
                try {
                    Socket client = serverSocket.accept();
                    server.pool.submit(() -> server.getHandler().accept(client));
                } catch (RejectedExecutionException ignored) {}
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}