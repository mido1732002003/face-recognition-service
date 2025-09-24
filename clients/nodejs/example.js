#!/usr/bin/env node
/**
 * Face Recognition Service Node.js Client Example
 * 
 * Usage:
 *   node example.js enroll <person_id> <image_path> [<image_path>...]
 *   node example.js identify <image_path>
 *   node example.js verify <person_id> <image_path>
 *   node example.js stats
 */

const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

class FaceRecognitionClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = apiKey ? { 'X-API-Key': apiKey } : {};
        this.timeout = 30000;
    }

    async enroll(personId, imagePaths, qualityThreshold = 0.5) {
        const formData = new FormData();
        
        // Add images
        for (const imagePath of imagePaths) {
            if (!fs.existsSync(imagePath)) {
                console.warn(`File not found: ${imagePath}`);
                continue;
            }
            
            formData.append('images', fs.createReadStream(imagePath), {
                filename: path.basename(imagePath),
                contentType: 'image/jpeg'
            });
        }
        
        // Add parameters
        formData.append('quality_threshold', qualityThreshold.toString());
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v1/enroll/${personId}`,
                formData,
                {
                    headers: {
                        ...this.headers,
                        ...formData.getHeaders()
                    },
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async identify(imagePath, similarityThreshold = 0.65, topK = 5) {
        if (!fs.existsSync(imagePath)) {
            return { error: `File not found: ${imagePath}` };
        }
        
        const formData = new FormData();
        formData.append('image', fs.createReadStream(imagePath), {
            filename: path.basename(imagePath),
            contentType: 'image/jpeg'
        });
        formData.append('similarity_threshold', similarityThreshold.toString());
        formData.append('top_k', topK.toString());
        formData.append('return_face_data', 'false');
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v1/identify`,
                formData,
                {
                    headers: {
                        ...this.headers,
                        ...formData.getHeaders()
                    },
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async verify(personId, imagePath, similarityThreshold = 0.65) {
        if (!fs.existsSync(imagePath)) {
            return { error: `File not found: ${imagePath}` };
        }
        
        const formData = new FormData();
        formData.append('image', fs.createReadStream(imagePath), {
            filename: path.basename(imagePath),
            contentType: 'image/jpeg'
        });
        formData.append('similarity_threshold', similarityThreshold.toString());
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v1/verify/${personId}`,
                formData,
                {
                    headers: {
                        ...this.headers,
                        ...formData.getHeaders()
                    },
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async getPerson(personId) {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/v1/persons/${personId}`,
                {
                    headers: this.headers,
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async listPersons(offset = 0, limit = 100) {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/v1/persons`,
                {
                    params: { offset, limit },
                    headers: this.headers,
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async getStats() {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/v1/stats`,
                {
                    headers: this.headers,
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }

    async healthCheck() {
        try {
            const response = await axios.get(
                `${this.baseUrl}/health`,
                {
                    headers: this.headers,
                    timeout: this.timeout
                }
            );
            
            return response.data;
        } catch (error) {
            return { error: error.message };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length < 1) {
        console.log('Usage:');
        console.log('  node example.js enroll <person_id> <image_path> [<image_path>...]');
        console.log('  node example.js identify <image_path>');
        console.log('  node example.js verify <person_id> <image_path>');
        console.log('  node example.js stats');
        process.exit(1);
    }
    
    // Initialize client
    const baseUrl = process.env.API_BASE_URL || 'http://localhost:8000';
    const apiKey = process.env.API_KEY || null;
    
    const client = new FaceRecognitionClient(baseUrl, apiKey);
    const command = args[0];
    
    try {
        switch (command) {
            case 'enroll':
                if (args.length < 3) {
                    console.error('Usage: node example.js enroll <person_id> <image_path> [<image_path>...]');
                    process.exit(1);
                }
                
                const personId = args[1];
                const imagePaths = args.slice(2);
                
                console.log(`Enrolling ${imagePaths.length} image(s) for ${personId}...`);
                const enrollResult = await client.enroll(personId, imagePaths);
                
                if (enrollResult.faces_enrolled !== undefined) {
                    console.log(`✅ Successfully enrolled ${enrollResult.faces_enrolled} face(s)`);
                } else {
                    console.log(`❌ Enrollment failed:`, enrollResult);
                }
                break;
                
            case 'identify':
                if (args.length < 2) {
                    console.error('Usage: node example.js identify <image_path>');
                    process.exit(1);
                }
                
                const identifyPath = args[1];
                console.log(`Identifying face in ${identifyPath}...`);
                const identifyResult = await client.identify(identifyPath);
                
                if (identifyResult.matches && identifyResult.matches.length > 0) {
                    const match = identifyResult.matches[0];
                    console.log(`✅ Identified as: ${match.person_id}`);
                    console.log(`   Similarity: ${match.similarity.toFixed(3)}`);
                } else if (identifyResult.matches) {
                    console.log('❌ No match found (Unknown person)');
                } else {
                    console.log('❌ Identification failed:', identifyResult);
                }
                break;
                
            case 'verify':
                if (args.length < 3) {
                    console.error('Usage: node example.js verify <person_id> <image_path>');
                    process.exit(1);
                }
                
                const verifyPersonId = args[1];
                const verifyPath = args[2];
                
                console.log(`Verifying if face belongs to ${verifyPersonId}...`);
                const verifyResult = await client.verify(verifyPersonId, verifyPath);
                
                if (verifyResult.verified !== undefined) {
                    if (verifyResult.verified) {
                        console.log(`✅ Verified: Face belongs to ${verifyPersonId}`);
                        console.log(`   Similarity: ${verifyResult.similarity.toFixed(3)}`);
                    } else {
                        console.log(`❌ Not verified: Face does not belong to ${verifyPersonId}`);
                    }
                } else {
                    console.log('❌ Verification failed:', verifyResult);
                }
                break;
                
            case 'stats':
                console.log('Getting system statistics...');
                const stats = await client.getStats();
                console.log(JSON.stringify(stats, null, 2));
                break;
                
            case 'health':
                console.log('Checking service health...');
                const health = await client.healthCheck();
                console.log(JSON.stringify(health, null, 2));
                break;
                
            default:
                console.error(`Unknown command: ${command}`);
                process.exit(1);
        }
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = { FaceRecognitionClient };