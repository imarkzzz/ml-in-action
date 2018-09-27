package mediator;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by mark on 2018/9/27.
 */
public class ChatRoom {
    private List<User> roomMates;

    public List<User> getRoomMates() {
        return roomMates;
    }

    public void setRoomMates(List<User> roomMates) {
        this.roomMates = roomMates;
    }

    public ChatRoom() {
        roomMates = new ArrayList<>();
    }

    public void showMessage(String message, User user) {
        for (User u: roomMates) {
            if(user != u) {
                u.getMessage(message);
            }
        }
    }
}
