package facade;

/**
 * Created by mark on 2018/10/26.
 */
public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Rectangle::draw()");
    }
}
