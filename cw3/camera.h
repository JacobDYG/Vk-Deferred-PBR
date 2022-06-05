#pragma once
#include <glm/glm.hpp>
#include <vector>


namespace camera
{
	enum keys
	{
		KEY_W,
		KEY_A,
		KEY_S,
		KEY_D,
		KEY_Q,
		KEY_E,
		KEY_SHIFT,
		KEY_CTRL,
		MOUSE_1,
		MOUSE_2
	};

	class Camera
	{
	public:
		Camera() noexcept = default;
		Camera(glm::vec3 aInitialPos);

		Camera(Camera& aOther) = delete;
		Camera& operator= (Camera const&) = delete;

		Camera(Camera&& aOther) noexcept;
		Camera& operator= (Camera&& aOther) noexcept;

	private:
		bool mKeyW = false;
		bool mKeyA = false;
		bool mKeyS = false;
		bool mKeyD = false;
		bool mKeyQ = false;
		bool mKeyE = false;
		bool mKeyShift = false;
		bool mKeyCtrl = false;
		bool mMouse1 = false;
		bool mMouse2 = false;

		float mSpeed = 3.0f;

	public:
		glm::vec3 mCameraPosition = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec3 mCameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
		glm::vec3 mCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

		bool getKeyState(int aKey);
		void setKeyState(int aKey, bool aState);

		float getSpeed() { return mSpeed; };
		void setSpeed(float aSpeed) { mSpeed = aSpeed; };

		void stepPosition(float aDeltaTime);
	};
};
