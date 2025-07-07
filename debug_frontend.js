// 模拟前端请求
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function testFrontendRequest() {
    console.log('🔍 测试前端请求格式...');
    
    // 创建一个简单的测试文件
    const testImageBuffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==', 'base64');
    fs.writeFileSync('test_frontend.png', testImageBuffer);
    
    const formData = new FormData();
    formData.append('file', fs.createReadStream('test_frontend.png'));
    formData.append('color_count', '12');
    formData.append('edge_enhancement', 'true');
    formData.append('noise_reduction', 'true');
    
    try {
        const response = await axios.post('http://localhost:8000/api/process', formData, {
            headers: {
                ...formData.getHeaders()
            }
        });
        
        console.log('✅ 成功:', response.status);
        console.log('📋 响应:', response.data);
    } catch (error) {
        console.log('❌ 错误:', error.response?.status);
        console.log('📋 错误详情:', error.response?.data);
    } finally {
        fs.unlinkSync('test_frontend.png');
    }
}

testFrontendRequest();
