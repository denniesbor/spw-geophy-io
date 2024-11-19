import axiosInstance from './axiosInstance';

export const login = async (username, password) => {
    const response = await axiosInstance.post('/api/token/', { username, password });
    const user = response.data;

    if (user.access) {
        localStorage.setItem('user', JSON.stringify(user));
    }
    return user;
};

export const logout = () => {
    localStorage.removeItem('user');
};

export const getCurrentUser = () => {
    return JSON.parse(localStorage.getItem('user'));
};

export const getUserDetails = async () => {
    try {
        const response = await axiosInstance.get('/gis/users/me/');
        return response.data;
    } catch (error) {
        console.error('Error fetching user details:', error);
        throw error;
    }
};
