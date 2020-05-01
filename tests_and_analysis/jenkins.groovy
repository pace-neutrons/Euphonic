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
    withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
            variable: 'api_token')]) {
        String GITHUB_API_BRANCH_URL = "https://api.github.com/repos/pace-neutrons/Euphonic/git/ref/heads/${env.JOB_BASE_NAME}"
        def branch_url_response = GITHUB_API_BRANCH_URL.toURL().getText(requestProperties: ['Authorization': "token ${api_token}"])
        def commit_url = new JsonSlurper().parseText(branch_url_response).object.url
        def commit_url_response = commit_url.toURL().getText(requestProperties: ['Authorization': "token ${api_token}"])
        return new JsonSlurper().parseText(commit_url_response).author.email
    }
}

return this
