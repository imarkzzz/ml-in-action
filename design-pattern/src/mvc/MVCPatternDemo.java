package mvc;

/**
 * Created by mark on 2018/10/24.
 */
public class MVCPatternDemo {
    public static void main(String[] args) {
        // 从数据库获取学生记录
        Student model = retriveStudentFromDatabase();

        // 创建一个试图：把学生详细信息输出到控制台
        StudentView view = new StudentView();

        StudentController controller = new StudentController(model, view);
        controller.updateView();

        // 更新模型数据
        controller.setStudentName("John");
        controller.updateView();
    }

    private static Student retriveStudentFromDatabase() {
        Student student = new Student();
        student.setName("Mark");
        student.setRollNo("0");
        return student;
    }
}
