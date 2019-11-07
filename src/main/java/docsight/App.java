package docsight;

import java.io.File;
import org.opencv.core.Core;
import org.apache.commons.io.FileUtils;

public final class App {
    private static final String SRC_FOLDER_PATH = "/home/alex/Desktop/windowsshare/ocr";

    public static void main(String[] args) throws Exception {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FileUtils.cleanDirectory(new File(SRC_FOLDER_PATH + "/text/"));
        FileUtils.cleanDirectory(new File(SRC_FOLDER_PATH + "/image/"));
        File folder = new File(SRC_FOLDER_PATH);
        for (File e : folder.listFiles()) {
            if (e.isFile()) {
                Document doc = new Document(e);
                System.out.println(doc.toString());
            }
        }
        System.out.println("[DONE]");
        System.exit(0);
    }
}
