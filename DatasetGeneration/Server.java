package DatasetGeneration;

import java.io.*;
import java.net.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class Server {

    private final AtomicInteger requestCount = new AtomicInteger();
    private final AtomicLong totalResponseTime = new AtomicLong();
    private final ThreadPoolExecutor pool;
    private final long startTime = System.currentTimeMillis();

    public Server(int poolSize) {
        pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(poolSize);
    }

    private class ClientHandler implements Runnable {
        private final Socket socket;
        ClientHandler(Socket socket) { this.socket = socket; }
        public void run() {
            long start = System.nanoTime();
            requestCount.incrementAndGet();
            try (PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {
                Thread.sleep(100); // simulate processing
                out.println("Hello from Server " + socket.getInetAddress());
            } catch (IOException | InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                totalResponseTime.addAndGet((System.nanoTime() - start) / 1_000_000);
                try { socket.close(); } catch (IOException ignored) {}
            }
        }
    }

    private class MetricsLogger implements Runnable {
        private final FileWriter writer;
        MetricsLogger(FileWriter writer) { this.writer = writer; }

        public void run() {
            try {
                long elapsed = System.currentTimeMillis() - startTime;
                int req = requestCount.getAndSet(0);
                long respTime = totalResponseTime.getAndSet(0);
                double avgRT = req > 0 ? respTime * 1.0 / req : 0;
                double mem = getMemoryUsage();
                int active = pool.getActiveCount();
                double util = active * 100.0 / pool.getMaximumPoolSize();
                double rps = req / 5.0;
                String level = req > 100 ? "High" : req > 50 ? "Medium" : "Low";

                writer.write(String.format("%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%s%n",
                        elapsed, req, 0.0, mem, avgRT, util, rps, level));
                writer.flush();

                System.out.printf("Time=%ds | Req=%d | Mem=%.1f%% | AvgRT=%.1fms | Threads=%d | Load=%s%n",
                        elapsed / 1000, req, mem, avgRT, active, level);
            } catch (IOException e) {
                System.err.println("Metrics error: " + e.getMessage());
            }
        }
    }

    private static double getMemoryUsage() {
        Runtime rt = Runtime.getRuntime();
        return (rt.totalMemory() - rt.freeMemory()) * 100.0 / rt.maxMemory();
    }

    public static void main(String[] args) {
        int port = 8010, poolSize = 50;
        Server server = new Server(poolSize);

        try (FileWriter writer = new FileWriter("enhanced_ml_dataset.csv", true)) {
            writer.write("Timestamp,Requests_Last_5s,CPU_Usage(%),Memory_Usage(%),Avg_Response_Time(ms),Thread_Utilization(%),Requests_Per_Second,Load_Level\n");
            System.out.println("Enhanced ML dataset: enhanced_ml_dataset.csv");

            ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
            scheduler.scheduleAtFixedRate(server.new MetricsLogger(writer), 0, 5, TimeUnit.SECONDS);

            try (ServerSocket serverSocket = new ServerSocket(port)) {
                serverSocket.setSoTimeout(3600000);
                System.out.println("Server listening on port " + port);
                System.out.println("Logging metrics every 5 seconds for ML training...");

                while (true) {
                    Socket socket = serverSocket.accept();
                    server.pool.submit(server.new ClientHandler(socket));
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                scheduler.shutdown();
                server.pool.shutdown();
                System.out.println("Server stopped gracefully");
            }
        } catch (IOException e) {
            System.err.println("File error: " + e.getMessage());
        }
    }
}