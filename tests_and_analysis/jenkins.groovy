def setGitHubBuildStatus(String status, String message, String context) {
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

def getGitCommitAuthorEmail() {
    script {
        withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
                variable: 'api_token')]) {
            if(isUnix()) {
                return sh(
                    script: """
                        commit_url="\$(\\
                            curl -s -H "Authorization: token ${api_token}" \\
                            --request GET https://api.github.com/repos/pace-neutrons/Euphonic/git/ref/heads/${env.JOB_BASE_NAME} \\
                            | jq ".object.url" | tr -d '"'\\
                        )" &&
                        echo "\$(\\
                            curl -s -H "Authorization: token ${api_token}" \\
                            --request GET \$commit_url |  jq '.author.email' | tr -d '"'\\
                        )"
                    """,
                    returnStdout: true
                )
            } else {
                error("Cannot get commit author in Windows")
            }
        }
    }
}

return this
