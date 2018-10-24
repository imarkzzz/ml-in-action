package servicelocator;

/**
 * Created by mark on 2018/10/24.
 */
public class ServiceLocator {
    private static Cache cache;

    static {
        cache = new Cache();
    }

    public static Service getService(String jndiname) {
        Service service = cache.getService(jndiname);

        if (service != null) {
            return service;
        }

        InitialContext context = new InitialContext();
        Service service1 = (Service) context.lookup(jndiname);
        cache.addService(service1);
        return service1;
    }
}
