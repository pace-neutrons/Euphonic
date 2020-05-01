def call(String status, String message, String context) {
    script {
        withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
                variable: 'api_token')]) {
            if (isUnix()) {
                sh """
                    curl -H "Authorization: token ${api_token}" \
                    --request POST \
                    --data '{ \
                        "state": "${status}", \
                        "description": "${message} on ${context}", \
                        "target_url": "$BUILD_URL", \
                        "context": "jenkins/${context}" \
                    }' \
                    https://api.github.com/repos/pace-neutrons/Euphonic/statuses/${env.GIT_COMMIT}
                """
            } else {
                powershell """
                    [Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"
                    \$payload = @{
                      "state" = "${status}";
                      "description" = "${message} on ${context}";
                      "target_url" = "$BUILD_URL";
                      "context" = "jenkins/${context}"}
                    Invoke-RestMethod -URI "https://api.github.com/repos/pace-neutrons/Euphonic/statuses/${env.GIT_COMMIT}" \
                      -Headers @{Authorization = "token ${api_token}"} \
                      -Method 'POST' \
                      -Body (\$payload|ConvertTo-JSON) \
                      -ContentType "application/json"
                  """
            }
        }
    }
}