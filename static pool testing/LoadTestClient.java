import java.io.*;
import java.net.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class LoadTestClient {
    private static final String SERVER_HOST = "127.0.0.1";
    private static final int SERVER_PORT = 8010;
    private static final int THREADS = 50;
    private static final int PHASE_REQUESTS = 100;
    private static final LoadPhase[] PHASE_SEQUENCE = { LoadPhase.LOW, LoadPhase.NORMAL, LoadPhase.HIGH };
    private static AtomicInteger totalSent = new AtomicInteger(0);
    private static AtomicInteger success = new AtomicInteger(0);
    private static AtomicInteger failed = new AtomicInteger(0);
    private static volatile boolean running = true;
    public static void main(String[] args) {
        System.out.println("\n==== Start LoadTestClient ====");
        System.out.printf("Server: %s:%d | Threads: %d\n", SERVER_HOST, SERVER_PORT, THREADS);
        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        // Shutdown hook for summary
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            running = false;
            executor.shutdownNow();
            printClientSummary();
        }));
        int phaseIdx = 0;
        while (running) {
            LoadPhase phase = PHASE_SEQUENCE[phaseIdx % PHASE_SEQUENCE.length];
            System.out.printf("\nPhase %d [%s]: Sending %d requests...\n", phaseIdx+1, phase, PHASE_REQUESTS);
            CountDownLatch latch = new CountDownLatch(PHASE_REQUESTS);
            int delay;
            if (phase == LoadPhase.HIGH) delay = 2;
            else if (phase == LoadPhase.NORMAL) delay = 12;
            else delay = 28;
            for (int i = 0; i < PHASE_REQUESTS && running; i++) {
                final int id = totalSent.incrementAndGet();
                executor.submit(() -> {
                    try {
                        sendRequest(id, phase);
                        success.incrementAndGet();
                    } catch (Exception e) {
                        failed.incrementAndGet();
                    } finally {
                        latch.countDown();
                    }
                });
                try { Thread.sleep(delay); } catch (InterruptedException ignored) {}
            }
            try { latch.await(); } catch (InterruptedException ignored) {}

            System.out.printf("Requests sent so far: %d | Successful: %d | Failed: %d\n",
                              totalSent.get(), success.get(), failed.get());
            phaseIdx++;
            try { Thread.sleep(350); } catch (InterruptedException ignored) {}
        }
    }
    private static void sendRequest(int id, LoadPhase phase) throws IOException {
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(SERVER_HOST, SERVER_PORT), 400);
            socket.setSoTimeout(1100);
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out.println("REQ#" + id + " PHASE=" + phase.name());
            String response = in.readLine();
            if (response == null || response.isEmpty())
                throw new IOException("Empty response for REQ#" + id);
            System.out.printf(" [Client] REQ#%d [%s]: server replied: %s\n", id, phase, response);
        }
    }
    private static void printClientSummary() {
        System.out.println("\n=== Client Session Performance Summary ===");
        System.out.printf("Total requests sent: %d\n", totalSent.get());
        System.out.printf("Successful: %d | Failed: %d\n", success.get(), failed.get());
        double failPct = (totalSent.get() == 0) ? 0 : failed.get() * 100.0 / totalSent.get();
        System.out.printf("Failure rate: %.2f%%\n", failPct);
        System.out.println("==========================================");
    }
}
