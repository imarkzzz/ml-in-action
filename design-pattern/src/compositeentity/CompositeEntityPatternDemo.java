package compositeentity;

/**
 * Created by mark on 2018/10/23.
 */
public class CompositeEntityPatternDemo {
    public static void main(String[] args) {
        Client client = new Client();
        client.setData("Test", "Data");
        client.printData();
        client.setData("Prod", "Log");
        client.printData();
    }
}
