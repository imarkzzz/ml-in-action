package state;

/**
 * Created by mark on 2018/9/27.
 */
public class StopState implements State {
    @Override
    public void doAction(Context context) {
        System.out.println("Player is in stop state!");
        context.setState(this);
    }

    @Override
    public String toString() {
        return "Stop State!22";
    }
}
