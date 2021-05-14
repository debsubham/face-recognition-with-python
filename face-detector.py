import cv2;
print("Completed");

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("rdj.jpg")


grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

cv2.imshow("Hello, I'm Subham Deb",img)
cv2.waitKey()

# cv2.imshow("Hello, I'm Subham Deb",grayscaled_img)
# cv2.waitKey()