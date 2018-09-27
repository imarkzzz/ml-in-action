package mediator;

/**
 * Created by mark on 2018/9/27.
 */
public class User {
    private String name;
    private ChatRoom chatRoom;

    public User(String name, ChatRoom chatRoom) {
        this.name = name;
        this.chatRoom = chatRoom;
        this.chatRoom.getRoomMates().add(this);
    }

    public void sendMessage(String message) {
        System.out.println(this.name + "说: " + message);
        chatRoom.showMessage(message, this);
    }

    public void getMessage(String message) {
        System.out.println(this.name + "收到消息: " + message);
    }
}
