package docsight;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.ArrayList;

import net.sourceforge.tess4j.Tesseract;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public final class Document {
    private static final String SRC_FOLDER_PATH = "/home/alex/Desktop/windowsshare/ocr";
    private static final String TESSDATA_FOLDER_PATH = "/home/alex/Desktop/windowsshare/docsight/alex/resources/tessdata";
    private static final String LANGUAGE = "eng";
    private File file;
    private boolean isText;
    private String content;

    public Document(File file) throws Exception {
        this.file = file;
        if(histogramTest() || textToAreaRatioTest()) {
            this.isText = true;
            Files.copy(this.file.toPath(), (new File(SRC_FOLDER_PATH + "/text/" + this.file.getName()).toPath()),
                    StandardCopyOption.REPLACE_EXISTING);
            //this.content = extractContent();
            //classifyContent();
        } else {
            this.isText = false;
            Files.copy(this.file.toPath(), (new File(SRC_FOLDER_PATH + "/image/" + this.file.getName())).toPath(),
                    StandardCopyOption.REPLACE_EXISTING);
        }
    }

    public boolean textToAreaRatioTest(){
        Mat img = Imgcodecs.imread(this.file.getAbsolutePath());
        if (img.empty()) {
            System.out.println("[ERROR] Cannot read image: " + this.file.getAbsolutePath());
            System.exit(0);
        }
        List<Rect> boundRect = new ArrayList<>();
        int imgArea = img.cols() * img.rows();
        double textArea = 0;
        Mat img_gray = new Mat(), img_sobel = new Mat(), img_threshold = new Mat(), element = new Mat(), hierarchy = new Mat();
        Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Sobel(img_gray, img_sobel, CvType.CV_8U, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);
        Imgproc.threshold(img_sobel, img_threshold, 0, 255, 8);
        //element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15,5));
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17,3));
        Imgproc.morphologyEx(img_threshold, img_threshold, Imgproc.MORPH_CLOSE, element);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img_threshold, contours, hierarchy, 0, 1);
        //List<MatOfPoint> contours_poly = new ArrayList<MatOfPoint>(contours.size());
        for (int i = 0; i < contours.size(); i++) {             
            MatOfPoint2f  mMOP2f1=new MatOfPoint2f();
            MatOfPoint2f  mMOP2f2=new MatOfPoint2f();
            contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
            //Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, 2, true);
            Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, 3, true);
            mMOP2f2.convertTo(contours.get(i), CvType.CV_32S);
                Rect appRect = Imgproc.boundingRect(contours.get(i));
                if (appRect.width > 1.66*appRect.height && appRect.area() <= imgArea / 2 && appRect.area() > 100) {
                    boundRect.add(appRect);
                }
        }
        for (int i = 0; i < boundRect.size(); ++i) {
            Imgproc.rectangle(img, boundRect.get(i).br(), boundRect.get(i).tl(), new Scalar(0, 255, 0), 2, 8, 0);
            textArea += boundRect.get(i).area();
        }
        double textRatio = textArea / imgArea;
        /*
        System.out.println(textRatio);
        HighGui.imshow( "res", img);
        HighGui.waitKey(0);
        */
        if (textRatio > 0.11)
            return true;
        return false;
    }

    public boolean histogramTest() throws Exception {
        Mat src = Imgcodecs.imread(this.file.getAbsolutePath());
        if (src.empty()) {
            System.out.println("[ERROR] Cannot read image: " + this.file.getAbsolutePath());
            System.exit(0);
        }
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(src, bgrPlanes);
        int histSize = 256;
        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);
        boolean accumulate = false;
        Mat bHist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), bHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), new Mat(), gHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), new Mat(), rHist, new MatOfInt(histSize), histRange, accumulate);
        int histW = 512, histH = 400;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );
        Core.normalize(bHist, bHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(gHist, gHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(rHist, rHist, 0, histImage.rows(), Core.NORM_MINMAX);
        float[] bHistData = new float[(int) (bHist.total() * bHist.channels())];
        bHist.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (gHist.total() * gHist.channels())];
        gHist.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (rHist.total() * rHist.channels())];
        rHist.get(0, 0, rHistData);
        int sides = 0, center = 0;
        for (int i = 0; i < histSize; ++i) {
            if (i < histSize / 4 || i >= 3 * histSize / 4)
                sides += (bHistData[i] + gHistData[i] + rHistData[i]);
            else
                center += (bHistData[i] + gHistData[i] + rHistData[i]);
        }
        /* histogram equalizer
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(src, src);
        Imgcodecs.imwrite(SRC_FOLDER_PATH + "/text/" + file.getName(), src);
        HighGui.imshow( "res", src);
        HighGui.waitKey(0);
        */
        for (int i = 1; i < histSize; ++i ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
        }
        /*
        HighGui.imshow( "Source image", src );
        HighGui.imshow( "calcHist Demo", histImage );
        HighGui.waitKey(0);
        */
        if (sides / 2 > center)
            return true;
        return false;
    }

    public String extractContent() {
        try {
            Tesseract tesseract = new Tesseract();
            tesseract.setLanguage(LANGUAGE);
            tesseract.setDatapath(TESSDATA_FOLDER_PATH);
            return tesseract.doOCR(this.file);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return null;
        }
    }

    public void classifyContent() {
        ArrayList<ArrayList<String>> checker = new ArrayList<ArrayList<String>>();
        ArrayList<String> bill = new ArrayList<>();
        bill.add("rechnung");
        checker.add(bill);
        ArrayList<String> papyrus = new ArrayList<>();
        papyrus.add("papyrus");
        checker.add(papyrus);
        for (int i = 0; i < checker.size(); ++i) {
            for (int j = 0; j < checker.get(i).size(); ++j) {
                if (this.content.toLowerCase().contains(checker.get(i).get(j)))
                    System.out.println("[" + checker.get(i).toString().toUpperCase() + "]");
            }
        }
    }

    @Override
    public String toString() {
        return "Document [content=" + content + ", file=" + file +  ", isText=" + isText + "]";
    }
}
