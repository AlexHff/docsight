package docsight;

import java.io.File;

import java.util.List;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public final class TextDetector {
    private File file;

    public TextDetector(File file, String srcPath) throws Exception {
        this.file = file;
    }

    public double textRatio() {
        Mat img = Imgcodecs.imread(this.file.getAbsolutePath());
        if (img.empty()) {
            System.out.println("[ERROR] Cannot read image: " + this.file.getAbsolutePath());
            System.exit(0);
        }
        List<Rect> boundRect = new ArrayList<>();
        int imgArea = img.cols() * img.rows();
        double textArea = 0;
        Mat img_gray = new Mat(), img_sobel = new Mat(), img_threshold = new Mat(), element = new Mat(),
                hierarchy = new Mat();
        Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Sobel(img_gray, img_sobel, CvType.CV_8U, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);
        Imgproc.threshold(img_sobel, img_threshold, 0, 255, 8);
        // element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15,5));
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17, 3));
        Imgproc.morphologyEx(img_threshold, img_threshold, Imgproc.MORPH_CLOSE, element);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img_threshold, contours, hierarchy, 0, 1);
        // List<MatOfPoint> contours_poly = new ArrayList<MatOfPoint>(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
            MatOfPoint2f mMOP2f2 = new MatOfPoint2f();
            contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
            // Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, 2, true);
            Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, 3, true);
            mMOP2f2.convertTo(contours.get(i), CvType.CV_32S);
            Rect appRect = Imgproc.boundingRect(contours.get(i));
            if (appRect.width > 1.66 * appRect.height && appRect.area() <= imgArea / 2 && appRect.area() > 100) {
                boundRect.add(appRect);
            }
        }
        for (int i = 0; i < boundRect.size(); ++i) {
            Imgproc.rectangle(img, boundRect.get(i).br(), boundRect.get(i).tl(), new Scalar(0, 255, 0), 2, 8, 0);
            textArea += boundRect.get(i).area();
        }
        double textRatio = textArea / imgArea;
        HighGui.imshow("res", img);
        HighGui.waitKey(0);
        return textRatio;
    }
}
