package dynamic;

import java.io.*;
import java.net.*;
import java.util.Scanner;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Server {
    private static final int MIN = 10;
    private static final int MAX = 160;
    private final ThreadPoolExecutor pool;
    private final AtomicLong requestCount = new AtomicLong();
    private PrintWriter csvLogger;

    public Server() {
        pool = new ThreadPoolExecutor(
                MIN, MAX, 60L, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(2000),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        pool.allowCoreThreadTimeOut(false);
        pool.prestartAllCoreThreads();

        try {
            csvLogger = new PrintWriter(new FileWriter("thread_analysis.csv", false), true);
            csvLogger.println("Timestamp,PoolSize,Active,Utilization,Queue,ReqDelta,ML_Suggest,Target");
        } catch (IOException e) {
            System.err.println("CSV Error: " + e.getMessage());
        }
    }

    private void handleClient(Socket client) {
        try (PrintWriter out = new PrintWriter(client.getOutputStream(), true)) {
            Thread.sleep(50);
            out.println("OK-" + requestCount.incrementAndGet());
        } catch (Exception ignored) {
        } finally {
            try { client.close(); } catch (IOException ignored) {}
        }
    }

    public void startMonitor() {
        new Thread(() -> {
            long lastCount = 0;
            while (true) {
                try {
                    Thread.sleep(5000);

                    long currentCount = requestCount.get();
                    long delta = currentCount - lastCount;
                    lastCount = currentCount;

                    sendMLUpdate(delta);
                    int mlSuggested = fetchMLSuggestion(delta);
                    int targetSize = calculateTarget(mlSuggested);

                    logMetrics(delta, mlSuggested, targetSize);
                    adjustPoolSize(targetSize);

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    private void sendMLUpdate(long count) {
        try {
            HttpURLConnection conn = (HttpURLConnection) new URI("http://localhost:5000/ml/update_load/" + count).toURL().openConnection();
            conn.setRequestMethod("POST");
            conn.setConnectTimeout(500);
            conn.getResponseCode();
            conn.disconnect();
        } catch (Exception ignored) {}
    }

    private int fetchMLSuggestion(long fallbackDelta) {
        try {
            HttpURLConnection conn = (HttpURLConnection) new URI("http://localhost:5000/ml/suggest_threads").toURL().openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(1000);

            try (Scanner sc = new Scanner(conn.getInputStream())) {
                String response = sc.useDelimiter("\\A").next();
                return parseJsonInt(response, "suggested_threads");
            }
        } catch (Exception e) {
            return Math.max(MIN, Math.min(MAX, (int)(fallbackDelta / 5.0)));
        }
    }

    private int calculateTarget(int mlSuggested) {
        int active = pool.getActiveCount();
        int currentCore = pool.getCorePoolSize();
        int queueSize = pool.getQueue().size();

        int target = (int)((active + 8) * 0.7 + mlSuggested * 0.3);

        if (queueSize > 500) {
            target = Math.max(target, currentCore + 30);
        } else if (queueSize > 100) {
            target = Math.max(target, currentCore + 15);
        }

        return Math.max(MIN, Math.min(MAX, target));
    }

    private void adjustPoolSize(int target) {
        int currentCore = pool.getCorePoolSize();
        if (Math.abs(target - currentCore) < 3) return;

        if (target > currentCore) {
            pool.setMaximumPoolSize(target);
            pool.setCorePoolSize(target);
        } else {
            pool.setCorePoolSize(target);
            pool.setMaximumPoolSize(target);
        }
    }

    private void logMetrics(long delta, int mlSuggested, int target) {
        if (csvLogger == null) return;

        int poolSize = pool.getPoolSize();
        int active = pool.getActiveCount();
        double util = poolSize > 0 ? (active * 100.0 / poolSize) : 0;

        csvLogger.printf("%d,%d,%d,%.2f,%d,%d,%d,%d%n",
                System.currentTimeMillis(), poolSize, active, util,
                pool.getQueue().size(), delta, mlSuggested, target);
    }

    private int parseJsonInt(String json, String key) {
        Matcher m = Pattern.compile("\"" + key + "\":\\s*(\\d+)").matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    public static void main(String[] args) {
        Server server = new Server();
        server.startMonitor();

        System.out.println("Server running on port 8010...");

        try (ServerSocket serverSocket = new ServerSocket(8010, 500)) {
            while (true) {
                try {
                    Socket client = serverSocket.accept();
                    server.pool.submit(() -> server.handleClient(client));
                } catch (RejectedExecutionException ignored) {}
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}