package docsight;

import java.io.File;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;

public final class App {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Too few arguments.");
            System.exit(0);
        } else if (args.length > 1) {
            System.err.println("Too many arguments.");
            System.exit(0);
        }
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FileUtils.cleanDirectory(new File(args[0] + "/text/"));
        FileUtils.cleanDirectory(new File(args[0] + "/image/"));
        File folder = new File(args[0]);
        for (File e : folder.listFiles()) {
            if (e.isFile()) {
                TextDetector instance = new TextDetector(e, args[0]);
                System.out.println(instance.textRatio());
            }
        }
        System.out.println("[DONE]");
        System.exit(0);
    }
}
