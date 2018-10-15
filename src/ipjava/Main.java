package ipjava;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;


//-Djava.library.path=/usr/java/packages/lib/P


public class Main {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main (String[] args) {

        String bookObject = "/home/evgeniy/Pictures/card.jpg";
        String bookScene = "/home/evgeniy/Pictures/cardScene.jpg";

        ObjectDetector bookDetector = new ObjectDetector(bookObject);
        bookDetector.setSceneImage(bookScene);
        Highgui.imwrite("/home/evgeniy/Pictures/result.jpg", bookDetector.detecObject());

    }
}

