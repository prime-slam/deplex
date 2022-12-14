file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/deplex)

# Copy pure-python code from SRC dir
file(COPY ${PYTHON_PACKAGE_SRC_DIR}/
        DESTINATION ${PYTHON_PACKAGE_DST_DIR})

# Copy Python-bindings
file(INSTALL ${PYTHON_BINARY_DIR}/
        DESTINATION ${PYTHON_PACKAGE_DST_DIR}/deplex)