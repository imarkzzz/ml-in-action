package mediator;

/**
 * Created by mark on 2018/9/27.
 */
public class MediatorPatternDemo {
    public static void main(String[] args) {
        ChatRoom chatRoom = new ChatRoom();
        User robert = new User("Robert", chatRoom);
        User john = new User("John", chatRoom);
        User Alice = new User("Alice", chatRoom);
        robert.sendMessage("Hi! John!");
        john.sendMessage("Hello! Robert!");
    }
}
