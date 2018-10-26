package facade;

/**
 * Created by mark on 2018/10/26.
 */
public class FacadePatternDemo {
    public static void main(String[] args) {
        ShapeMaker shapeMaker = new ShapeMaker();

        shapeMaker.drawCicle();
        shapeMaker.drawRectangle();
        shapeMaker.drawSquare();
    }
}
