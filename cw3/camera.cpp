#include "camera.h"

namespace camera
{
	Camera::Camera(glm::vec3 initialPos)
	{
		mCameraPosition = initialPos;
	}

	Camera::Camera(Camera&& aOther) noexcept :
		mCameraPosition(std::move(aOther.mCameraPosition)),
		mCameraFront(std::move(aOther.mCameraFront)),
		mCameraUp(std::move(aOther.mCameraUp))
	{
		mKeyW = aOther.mKeyW;
		mKeyA = aOther.mKeyA;
		mKeyS = aOther.mKeyS;
		mKeyD = aOther.mKeyD;
		mKeyQ = aOther.mKeyQ;
		mKeyE = aOther.mKeyE;
		mKeyShift = aOther.mKeyShift;
		mKeyCtrl = aOther.mKeyCtrl;
	}

	Camera& Camera::operator=(Camera&& aOther) noexcept
	{
		return std::move(aOther);
	}

	bool Camera::getKeyState(int key)
	{
		switch (key)
		{
		case KEY_W:
			return mKeyW;
		case KEY_S:
			return mKeyS;
		case KEY_A:
			return mKeyA;
		case KEY_D:
			return mKeyD;
		case KEY_Q:
			return mKeyQ;
		case KEY_E:
			return mKeyE;
		case KEY_SHIFT:
			return mKeyShift;
		case KEY_CTRL:
			return mKeyCtrl;
		case MOUSE_1:
			return mMouse1;
		case MOUSE_2:
			return mMouse2;
		default:
			break;
		}
		return false;
	}

	void Camera::setKeyState(int key, bool state)
	{
		switch (key)
		{
		case KEY_W:
			mKeyW = state;
			break;
		case KEY_S:
			mKeyS = state;
			break;
		case KEY_A:
			mKeyA = state;
			break;
		case KEY_D:
			mKeyD = state;
			break;
		case KEY_Q:
			mKeyQ = state;
			break;
		case KEY_E:
			mKeyE = state;
			break;
		case KEY_SHIFT:
			mKeyShift = state;
			break;
		case KEY_CTRL:
			mKeyCtrl = state;
			break;
		case MOUSE_1:
			mMouse1 = state;
			break;
		case MOUSE_2:
			mMouse2 = state;
			break;
		default:
			break;
		}
	}
	void Camera::stepPosition(float aDeltaTime)
	{
		float cameraSpeed = aDeltaTime * mSpeed;
		if (mKeyShift)
			cameraSpeed *= 2.0f;
		if (mKeyCtrl)
			cameraSpeed /= 2.0f;
		glm::vec3 cameraCross = glm::normalize(glm::cross(mCameraFront, mCameraUp));
		if (mKeyW)
		{
			mCameraPosition += cameraSpeed * mCameraFront;
		}
		if (mKeyS)
		{
			mCameraPosition -= cameraSpeed * mCameraFront;
		}
		if (mKeyA)
		{
			mCameraPosition -= cameraCross * cameraSpeed;
		}
		if (mKeyD)
		{
			mCameraPosition += cameraCross * cameraSpeed;
		}
		if (mKeyE)
		{
			mCameraPosition += mCameraUp * cameraSpeed;
		}
		if (mKeyQ)
		{
			mCameraPosition -= mCameraUp * cameraSpeed;
		}
	}
}
