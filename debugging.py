import cv2

# Load apple image
apple = cv2.imread('images/apple.jpg')
size = 100
apple = cv2.resize(apple, (size, size))

# Create a mask of the logo
img2gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

# # Show the original apple image
# cv2.imshow('Original Apple', apple)
print(mask)
print(mask.shape)
# Show the mask
cv2.imshow('Mask', mask)

# Wait for a key press to close the windows
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
