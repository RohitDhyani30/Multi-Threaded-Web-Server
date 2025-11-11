package testing_all.dynamic_testing.server;

import java.lang.management.ManagementFactory;
import com.sun.management.OperatingSystemMXBean;

public class SysMetrics {
    private static final OperatingSystemMXBean os =
            (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    private static final Runtime rt = Runtime.getRuntime();

    public static double cpu() {
        double v = os.getProcessCpuLoad();
        return v < 0 ? 0 : v * 100.0;
    }

    public static long ramUsed() {
        return rt.totalMemory() - rt.freeMemory();
    }

    public static long ramTotal() {
        return rt.totalMemory();
    }
}
