package ipjava;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

import java.util.LinkedList;
import java.util.List;

public class ObjectDetector {
    private final Mat objectImage; //object to search
    private Mat sceneImage = null;  //scene to search
    private MatOfKeyPoint objectDescriptors;
    private KeyPoint[] keyPoints;

    private final FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
    private final DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

    private Mat outputImage;  // the matrix for output image.

    private MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();

    ObjectDetector(Mat objectImage) {
        this.objectImage = objectImage;
        detectKeyPoints();
    }

    ObjectDetector(String pathToObjectImage) {
        this.objectImage = Highgui.imread(pathToObjectImage, Highgui.CV_LOAD_IMAGE_COLOR);
        detectKeyPoints();
    }

    /**
     * Set image, where we should find object
     * @param sceneImage - scene to searching
     * @see ObjectDetector#setSceneImage(Mat)
     */
    public void setSceneImage(Mat sceneImage) {
        this.sceneImage = sceneImage;
    }

    /**
     * Set image, where we should find object
     * @param pathToSceneImage - scene to searching
     * @see ObjectDetector#setSceneImage(Mat)
     */
    public void setSceneImage(String pathToSceneImage) {
        this.sceneImage = Highgui.imread(pathToSceneImage, Highgui.CV_LOAD_IMAGE_COLOR);
    }

    public Mat detecObject(){
        if (sceneImage == null){
            return null;
        }
        return matchingWithScene();
    }

    private void detectKeyPoints(){
        objectKeyPoints = new MatOfKeyPoint();
        featureDetector.detect(objectImage, objectKeyPoints);
        keyPoints = objectKeyPoints.toArray();

        objectDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

        outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar newKeypointColor = new Scalar(255, 0, 0);

        Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

    }

    private Mat matchingWithScene(){
        // Match object image with the scene image
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();

        // Detecting key points in background image
        featureDetector.detect(sceneImage, sceneKeyPoints);

        // Computing descriptors in background image
        descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);

//        Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
//        Scalar matchestColor = new Scalar(0, 255, 0);

        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);

        // Matching object and scene images
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);

        // Calculating good match list
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

        float nndrRatio = 0.7f;

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);

            }
        }

        if (goodMatchesList.size() >= 7) { //object found

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();

            for (int i = 0; i < goodMatchesList.size(); i++) {
                objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
                scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
            }

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{objectImage.cols(), 0});
            obj_corners.put(2, 0, new double[]{objectImage.cols(), objectImage.rows()});
            obj_corners.put(3, 0, new double[]{0, objectImage.rows()});

            // Transforming object corners to scene corners
            Core.perspectiveTransform(obj_corners, scene_corners, homography);

            Mat img = sceneImage.clone();

            Core.line(img, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);

            // Drawing matches image

            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);

//            Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);

//            Highgui.imwrite("/home/evgeniy/Pictures/outputImage.jpg", outputImage);
//            Highgui.imwrite("/home/evgeniy/Pictures/matchoutput.jpg", matchoutput);
//            Highgui.imwrite("/home/evgeniy/Pictures/img.jpg", img);

            return img;
        } else {
            return null;
        }
    }
}
