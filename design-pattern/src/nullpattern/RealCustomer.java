package nullpattern;

import com.sun.org.apache.regexp.internal.RE;

/**
 * Created by mark on 2018/10/24.
 */
public class RealCustomer extends AbstractCustomer {

    public RealCustomer(String name) {
        this.name = name;
    }

    @Override
    public boolean isNil() {
        return false;
    }

    @Override
    public String getName() {
        return name;
    }
}
