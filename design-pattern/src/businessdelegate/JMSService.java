package businessdelegate;

/**
 * Created by mark on 2018/10/26.
 */
public class JMSService implements BusinessService {
    @Override
    public void doProcessing() {
        System.out.println("Processing task by invoking JMS Service");
    }
}
